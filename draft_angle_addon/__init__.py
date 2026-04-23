bl_info = {
    "name": "Draft Angle",
    "author": "bebelllm",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Draft Angle",
    "description": "Apply manufacturing draft angle to mesh walls (hinge-based algorithm + Laplacian smoothing)",
    "category": "Mesh",
}

import bpy
import bmesh
import mathutils
import numpy as np
from collections import defaultdict
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty,
    EnumProperty, PointerProperty,
)
from bpy.types import Operator, Panel, PropertyGroup


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class DRAFT_Settings(PropertyGroup):

    draft_angle: FloatProperty(
        name="Draft Angle",
        description="Minimum draft angle to enforce on near-vertical walls (degrees)",
        default=2.0, min=0.1, max=45.0, step=10, precision=1,
        subtype="ANGLE",
        unit="ROTATION",
    )
    pull_axis: EnumProperty(
        name="Pull Direction",
        description="Mold release direction",
        items=[
            ("+Z", "+Z (up)",    "Release upward along Z"),
            ("-Z", "-Z (down)",  "Release downward along Z"),
            ("+Y", "+Y",         "Release along +Y"),
            ("-Y", "-Y",         "Release along -Y"),
            ("+X", "+X",         "Release along +X"),
            ("-X", "-X",         "Release along -X"),
        ],
        default="+Z",
    )
    vert_threshold: FloatProperty(
        name="Wall Threshold",
        description="Faces whose angle from pull direction exceeds this value are treated as walls",
        default=60.0, min=10.0, max=89.0, step=100, precision=0,
        subtype="ANGLE",
        unit="ROTATION",
    )
    smooth_iter: IntProperty(
        name="Smooth Iterations",
        description="Laplacian XY smoothing passes after correction (reduces sharp transitions)",
        default=5, min=0, max=30,
    )
    smooth_factor: FloatProperty(
        name="Smooth Factor",
        description="Strength of each Laplacian pass (0=none, 1=full)",
        default=0.4, min=0.0, max=1.0,
    )
    create_copy: BoolProperty(
        name="Create Copy",
        description="Apply draft on a duplicate; keep the original unchanged",
        default=True,
    )
    hide_source: BoolProperty(
        name="Hide Source",
        description="Hide the original object after applying (only relevant when Create Copy is on)",
        default=True,
    )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def _pull_vector(axis_str):
    mapping = {
        "+Z": (0, 0,  1),
        "-Z": (0, 0, -1),
        "+Y": (0,  1, 0),
        "-Y": (0, -1, 0),
        "+X": ( 1, 0, 0),
        "-X": (-1, 0, 0),
    }
    return mathutils.Vector(mapping[axis_str])


def apply_draft_angle(mesh_obj, draft_deg, pull_axis_str,
                      vert_threshold_deg, smooth_iter, smooth_factor):
    """
    Rigorous hinge-based draft angle algorithm.

    For each near-vertical face:
      - The topmost edge (relative to pull direction) is the *hinge* and stays fixed.
      - Every lower vertex is displaced outward (along the face's XY normal) by
            delta = (z_hinge - z_v) * (tan(DRAFT) - tan(current_draft))
        so that the face reaches exactly DRAFT degrees after correction.
      - When a vertex belongs to multiple corrected faces, the largest displacement wins.

    A final Laplacian XY pass (Z untouched) blends sharp transitions.
    """
    pull      = _pull_vector(pull_axis_str)
    draft_tan = np.tan(np.radians(draft_deg))
    cos_thresh = np.cos(np.radians(vert_threshold_deg))

    me = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # ── Pass 1 : compute per-vertex displacement ──────────────
    vert_disp = defaultdict(lambda: mathutils.Vector((0.0, 0.0)))

    for f in bm.faces:
        n = f.normal
        cos_a = min(max(abs(n.dot(pull)), 0.0), 1.0)

        if cos_a > cos_thresh:
            continue

        angle_from_pull = np.degrees(np.arccos(cos_a))
        current_draft   = 90.0 - angle_from_pull

        if current_draft >= draft_deg:
            continue

        outward_xy = mathutils.Vector((n.x, n.y, 0.0))
        if outward_xy.length < 1e-6:
            continue
        outward_xy.normalize()

        face_verts = list(f.verts)
        z_hinge    = max(v.co.dot(pull) for v in face_verts)

        current_tan = np.tan(np.radians(max(current_draft, 0.0)))
        delta_tan   = draft_tan - current_tan
        if delta_tan <= 0:
            continue

        for v in face_verts:
            dz = z_hinge - v.co.dot(pull)
            if dz < 1e-8:
                continue
            needed = outward_xy * (dz * delta_tan)
            if needed.length > vert_disp[v.index].length:
                vert_disp[v.index] = needed

    # ── Pass 2 : apply ───────────────────────────────────────
    modified = set()
    for v in bm.verts:
        if v.index in vert_disp:
            d = vert_disp[v.index]
            v.co.x += d.x
            v.co.y += d.y
            modified.add(v.index)

    # ── Pass 3 : Laplacian XY smoothing on modified verts ────
    for _ in range(smooth_iter):
        for v in bm.verts:
            if v.index not in modified:
                continue
            neighbours = [e.other_vert(v) for e in v.link_edges]
            if not neighbours:
                continue
            avg_x = sum(nb.co.x for nb in neighbours) / len(neighbours)
            avg_y = sum(nb.co.y for nb in neighbours) / len(neighbours)
            v.co.x += smooth_factor * (avg_x - v.co.x)
            v.co.y += smooth_factor * (avg_y - v.co.y)

    bm.to_mesh(me)
    me.update()
    bm.free()

    return len(modified)


def count_violations(mesh_obj, draft_deg, pull_axis_str):
    pull  = _pull_vector(pull_axis_str)
    bm    = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.faces.ensure_lookup_table()
    total = len(bm.faces)
    viol  = sum(
        1 for f in bm.faces
        if (90.0 - np.degrees(np.arccos(min(max(abs(f.normal.dot(pull)), 0.0), 1.0)))) < draft_deg
    )
    bm.free()
    return viol, total


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class DRAFT_OT_apply(Operator):
    bl_idname      = "draft.apply_angle"
    bl_label       = "Apply Draft Angle"
    bl_description = "Enforce a minimum draft angle on near-vertical walls"
    bl_options     = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        s         = context.scene.draft_settings
        src_obj   = context.active_object
        draft_deg = np.degrees(s.draft_angle)

        if s.create_copy:
            bpy.ops.object.select_all(action="DESELECT")
            src_obj.select_set(True)
            context.view_layer.objects.active = src_obj
            bpy.ops.object.duplicate()
            work_obj      = context.active_object
            work_obj.name = src_obj.name + f"_draft{draft_deg:.0f}deg"
            bpy.ops.object.convert(target="MESH")
            if s.hide_source:
                src_obj.hide_set(True)
                src_obj.hide_render = True
        else:
            work_obj = src_obj
            bpy.ops.object.convert(target="MESH")

        n_moved = apply_draft_angle(
            mesh_obj           = work_obj,
            draft_deg          = draft_deg,
            pull_axis_str      = s.pull_axis,
            vert_threshold_deg = np.degrees(s.vert_threshold),
            smooth_iter        = s.smooth_iter,
            smooth_factor      = s.smooth_factor,
        )

        n_viol, n_total = count_violations(work_obj, draft_deg, s.pull_axis)
        pct_ok = 100.0 * (1.0 - n_viol / max(n_total, 1))

        self.report(
            {"INFO"},
            f"Draft {draft_deg:.0f}° — {n_moved} verts moved, "
            f"{pct_ok:.1f}% faces compliant ({n_total - n_viol}/{n_total})",
        )
        return {"FINISHED"}


class DRAFT_OT_analyse(Operator):
    bl_idname      = "draft.analyse"
    bl_label       = "Analyse Violations"
    bl_description = "Count faces violating the current draft angle (no geometry change)"
    bl_options     = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        s         = context.scene.draft_settings
        obj       = context.active_object
        draft_deg = np.degrees(s.draft_angle)

        n_viol, n_total = count_violations(obj, draft_deg, s.pull_axis)
        pct_viol = 100.0 * n_viol / max(n_total, 1)

        self.report(
            {"INFO"},
            f"'{obj.name}': {n_viol}/{n_total} faces violate {draft_deg:.0f}° "
            f"({pct_viol:.1f}% violations, {100-pct_viol:.1f}% OK)",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class DRAFT_PT_panel(Panel):
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Draft Angle"
    bl_label       = "Draft Angle"

    def draw(self, context):
        layout = self.layout
        s      = context.scene.draft_settings

        box = layout.box()
        box.label(text="Parameters", icon="DRIVER_ROTATIONAL_DIFFERENCE")
        col = box.column(align=True)
        col.prop(s, "pull_axis")
        col.prop(s, "draft_angle")
        col.prop(s, "vert_threshold")

        box2 = layout.box()
        box2.label(text="Smoothing", icon="MOD_SMOOTH")
        col2 = box2.column(align=True)
        col2.prop(s, "smooth_iter")
        col2.prop(s, "smooth_factor", slider=True)

        box3 = layout.box()
        box3.label(text="Output", icon="OBJECT_DATA")
        col3 = box3.column(align=True)
        col3.prop(s, "create_copy")
        sub = col3.row()
        sub.enabled = s.create_copy
        sub.prop(s, "hide_source")

        layout.separator()
        layout.operator(DRAFT_OT_analyse.bl_idname, icon="VIEWZOOM")
        layout.operator(DRAFT_OT_apply.bl_idname,   icon="CHECKMARK")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
_classes = (
    DRAFT_Settings,
    DRAFT_OT_apply,
    DRAFT_OT_analyse,
    DRAFT_PT_panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.draft_settings = PointerProperty(type=DRAFT_Settings)


def unregister():
    del bpy.types.Scene.draft_settings
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
