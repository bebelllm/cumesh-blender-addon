bl_info = {
    "name": "CuMesh Remesh",
    "author": "mikab",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > CuMesh",
    "description": "GPU narrow-band dual-contouring remesh via CuMesh (external Python subprocess)",
    "category": "Mesh",
}

import os
import sys
import tempfile
import subprocess
import shutil

import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, PointerProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup


WORKER_FILENAME = "cumesh_worker.py"


def _worker_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKER_FILENAME)


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------
class CUMESH_AddonPreferences(AddonPreferences):
    bl_idname = __name__

    python_exe: StringProperty(
        name="External Python",
        description="Path to a Python interpreter (venv) with torch + cumesh installed",
        subtype="FILE_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="External Python interpreter that has torch + cumesh installed:")
        col.prop(self, "python_exe")
        col.separator()
        col.label(text="Example: C:/envs/cumesh/Scripts/python.exe", icon="INFO")
        col.label(text=f"Worker script: {_worker_path()}", icon="FILE_SCRIPT")


# ---------------------------------------------------------------------------
# Per-scene settings
# ---------------------------------------------------------------------------
class CUMESH_Settings(PropertyGroup):
    resolution: IntProperty(
        name="Resolution",
        description="Dual-contouring grid resolution along the longest bbox axis",
        default=128, min=16, max=1024,
    )
    band: IntProperty(
        name="Band",
        description="Narrow band width in voxels",
        default=1, min=1, max=8,
    )
    project_back: FloatProperty(
        name="Project Back",
        description="Projection strength of DC vertices back onto the original surface (0..1)",
        default=0.9, min=0.0, max=1.0,
    )
    scale_padding: FloatProperty(
        name="Scale Padding",
        description="Multiplier applied to the bbox extent to pad the DC volume",
        default=1.05, min=1.0, max=2.0,
    )
    hide_source: BoolProperty(
        name="Hide Source",
        description="Hide the original object after remeshing",
        default=True,
    )
    verbose: BoolProperty(
        name="Verbose",
        description="Print CuMesh progress to the system console",
        default=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_python(prefs):
    raw_py = (prefs.python_exe or "").strip()
    py_exe = bpy.path.abspath(raw_py) if raw_py else ""
    if not raw_py:
        return None, "CuMesh: 'External Python' is empty. Edit > Preferences > Add-ons > CuMesh Remesh > set python.exe of your torch+cumesh venv."
    if not os.path.isfile(py_exe):
        return None, f"CuMesh: External Python not found at: {py_exe}"
    worker = _worker_path()
    if not os.path.isfile(worker):
        return None, f"Worker script not found: {worker}"
    return (py_exe, worker), None


def _run_pipeline(context, src_obj, settings, py_exe, worker):
    """Run the cumesh subprocess on src_obj's evaluated mesh.

    Returns (mesh_datablock, error_message). On success error_message is None
    and mesh_datablock is a fresh bpy.data.meshes entry holding the remeshed
    geometry. On failure mesh_datablock is None.
    """
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = src_obj.evaluated_get(depsgraph)

    tmpdir = tempfile.mkdtemp(prefix="cumesh_")
    in_path = os.path.join(tmpdir, "input.ply")
    out_path = os.path.join(tmpdir, "output.ply")

    prev_active = context.view_layer.objects.active
    prev_selected = list(context.selected_objects)

    tmp_obj = None
    tmp_mesh = None
    imported_obj = None
    imported_mesh = None

    try:
        # Build a temp object holding the evaluated mesh (modifiers applied).
        # Keep matrix_world at identity so the PLY exporter writes LOCAL-space
        # vertex coords (the exporter bakes matrix_world into the coordinates
        # when apply_modifiers=True). The result object then carries the
        # source transform, avoiding double-transform on import.
        bpy.ops.object.select_all(action="DESELECT")
        tmp_mesh = bpy.data.meshes.new_from_object(eval_obj)
        tmp_obj = bpy.data.objects.new("_cumesh_tmp_in", tmp_mesh)
        context.collection.objects.link(tmp_obj)
        # Identity matrix on purpose (do NOT copy src_obj.matrix_world here)
        tmp_obj.select_set(True)
        context.view_layer.objects.active = tmp_obj

        bpy.ops.wm.ply_export(
            filepath=in_path,
            export_selected_objects=True,
            apply_modifiers=True,
            export_uv=False,
            export_normals=False,
            export_colors="NONE",
            ascii_format=False,
            forward_axis="Y",
            up_axis="Z",
        )

        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        tmp_obj = None
        bpy.data.meshes.remove(tmp_mesh, do_unlink=True)
        tmp_mesh = None

        cmd = [
            py_exe, worker,
            "--input", in_path,
            "--output", out_path,
            "--resolution", str(settings.resolution),
            "--band", str(settings.band),
            "--project-back", str(settings.project_back),
            "--scale-padding", str(settings.scale_padding),
        ]
        if settings.verbose:
            cmd.append("--verbose")

        print("[CuMesh] Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
        except subprocess.TimeoutExpired:
            return None, "CuMesh worker timed out (30 min)."

        if proc.stdout:
            print("[CuMesh stdout]\n" + proc.stdout)
        if proc.stderr:
            print("[CuMesh stderr]\n" + proc.stderr)

        if proc.returncode != 0:
            return None, f"CuMesh worker failed (code {proc.returncode}). See system console."
        if not os.path.isfile(out_path):
            return None, "Worker finished but no output file was produced."

        # Import result; this creates a new object we'll throw away after
        # transferring its mesh datablock.
        bpy.ops.object.select_all(action="DESELECT")
        before = set(bpy.data.objects)
        bpy.ops.wm.ply_import(filepath=out_path, forward_axis="Y", up_axis="Z")
        new_objs = [o for o in bpy.data.objects if o not in before]
        if not new_objs:
            return None, "PLY import did not create an object."
        imported_obj = new_objs[0]
        imported_mesh = imported_obj.data

        # Detach the mesh from the temp imported object so we can hand it back.
        imported_obj.data = bpy.data.meshes.new("_cumesh_empty")
        bpy.data.objects.remove(imported_obj, do_unlink=True)
        imported_obj = None

        return imported_mesh, None

    finally:
        # Cleanup temp scene objects if anything was left dangling
        if tmp_obj is not None:
            try: bpy.data.objects.remove(tmp_obj, do_unlink=True)
            except Exception: pass
        if tmp_mesh is not None:
            try: bpy.data.meshes.remove(tmp_mesh, do_unlink=True)
            except Exception: pass
        if imported_obj is not None:
            try: bpy.data.objects.remove(imported_obj, do_unlink=True)
            except Exception: pass
        # Restore selection
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        for o in prev_selected:
            try: o.select_set(True)
            except ReferenceError: pass
        if prev_active:
            try: context.view_layer.objects.active = prev_active
            except Exception: pass
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _store_params(obj, settings):
    obj["cumesh_resolution"] = int(settings.resolution)
    obj["cumesh_band"] = int(settings.band)
    obj["cumesh_project_back"] = float(settings.project_back)
    obj["cumesh_scale_padding"] = float(settings.scale_padding)


def _is_cumesh_result(obj):
    return obj is not None and obj.type == "MESH" and obj.cumesh_source is not None


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class CUMESH_OT_remesh(Operator):
    bl_idname = "cumesh.remesh"
    bl_label = "CuMesh Remesh"
    bl_description = "Remesh the active mesh using CuMesh narrow-band dual contouring"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and not _is_cumesh_result(obj)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        settings = context.scene.cumesh_settings

        resolved, err = _resolve_python(prefs)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        py_exe, worker = resolved

        src_obj = context.active_object
        new_mesh, err = _run_pipeline(context, src_obj, settings, py_exe, worker)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        new_mesh.name = src_obj.name + "_cumesh_mesh"
        new_obj = bpy.data.objects.new(src_obj.name + "_cumesh", new_mesh)
        context.collection.objects.link(new_obj)
        new_obj.matrix_world = src_obj.matrix_world.copy()

        # Link result -> source (this makes it a "live" non-destructive remesh)
        new_obj.cumesh_source = src_obj
        _store_params(new_obj, settings)

        if settings.hide_source:
            src_obj.hide_set(True)
            src_obj.hide_render = True

        bpy.ops.object.select_all(action="DESELECT")
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj

        self.report({"INFO"}, f"CuMesh remesh done: {new_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_refresh(Operator):
    bl_idname = "cumesh.refresh"
    bl_label = "Refresh CuMesh"
    bl_description = "Re-run CuMesh on the linked source and replace this object's mesh in-place"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _is_cumesh_result(context.active_object)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        settings = context.scene.cumesh_settings

        resolved, err = _resolve_python(prefs)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        py_exe, worker = resolved

        result_obj = context.active_object
        src_obj = result_obj.cumesh_source
        if src_obj is None or src_obj.type != "MESH":
            self.report({"ERROR"}, "Linked source is missing or not a mesh.")
            return {"CANCELLED"}

        new_mesh, err = _run_pipeline(context, src_obj, settings, py_exe, worker)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        # Swap in the new mesh datablock; remove the old one to avoid leak.
        old_mesh = result_obj.data
        new_mesh.name = old_mesh.name  # preserve naming
        result_obj.data = new_mesh
        try:
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh, do_unlink=True)
        except Exception:
            pass

        # Keep result aligned to source transform
        result_obj.matrix_world = src_obj.matrix_world.copy()
        _store_params(result_obj, settings)

        self.report({"INFO"}, f"CuMesh refreshed: {result_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_toggle_source(Operator):
    bl_idname = "cumesh.toggle_source"
    bl_label = "Toggle Source / Remesh"
    bl_description = "Swap visibility between the source mesh and its CuMesh result"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            return False
        if _is_cumesh_result(obj):
            return True
        # If active is a source, look for any result pointing to it
        for o in bpy.data.objects:
            if o.type == "MESH" and o.cumesh_source == obj:
                return True
        return False

    def execute(self, context):
        obj = context.active_object
        if _is_cumesh_result(obj):
            result = obj
            source = obj.cumesh_source
        else:
            source = obj
            result = next(
                (o for o in bpy.data.objects if o.type == "MESH" and o.cumesh_source == source),
                None,
            )
            if result is None:
                self.report({"ERROR"}, "No CuMesh result linked to this source.")
                return {"CANCELLED"}

        # Swap hide states
        src_hidden = source.hide_get()
        res_hidden = result.hide_get()
        source.hide_set(not src_hidden if src_hidden == res_hidden else res_hidden)
        result.hide_set(not res_hidden if src_hidden == res_hidden else src_hidden)
        # Make the now-visible one active
        if not source.hide_get():
            bpy.ops.object.select_all(action="DESELECT")
            source.select_set(True)
            context.view_layer.objects.active = source
        else:
            bpy.ops.object.select_all(action="DESELECT")
            result.select_set(True)
            context.view_layer.objects.active = result
        return {"FINISHED"}


class CUMESH_OT_select_source(Operator):
    bl_idname = "cumesh.select_source"
    bl_label = "Edit Source"
    bl_description = "Show and select the linked source mesh (so you can edit it before refreshing)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _is_cumesh_result(context.active_object)

    def execute(self, context):
        result = context.active_object
        src = result.cumesh_source
        if src is None:
            self.report({"ERROR"}, "No linked source.")
            return {"CANCELLED"}
        src.hide_set(False)
        src.hide_render = False
        bpy.ops.object.select_all(action="DESELECT")
        src.select_set(True)
        context.view_layer.objects.active = src
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class CUMESH_PT_panel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CuMesh"
    bl_label = "CuMesh Remesh"

    def draw(self, context):
        layout = self.layout
        s = context.scene.cumesh_settings
        obj = context.active_object

        col = layout.column(align=True)
        col.prop(s, "resolution")
        col.prop(s, "band")
        col.prop(s, "project_back", slider=True)
        col.prop(s, "scale_padding")

        layout.separator()
        layout.prop(s, "hide_source")
        layout.prop(s, "verbose")

        layout.separator()

        if _is_cumesh_result(obj):
            box = layout.box()
            box.label(text="Linked CuMesh Result", icon="LINKED")
            src = obj.cumesh_source
            row = box.row()
            row.label(text=f"Source: {src.name if src else '<missing>'}", icon="OBJECT_DATA")
            box.operator(CUMESH_OT_refresh.bl_idname, icon="FILE_REFRESH")
            row = box.row(align=True)
            row.operator(CUMESH_OT_select_source.bl_idname, icon="EDITMODE_HLT")
            row.operator(CUMESH_OT_toggle_source.bl_idname, icon="HIDE_OFF")

            # Show baked params for transparency
            if "cumesh_resolution" in obj:
                sub = box.column(align=True)
                sub.scale_y = 0.85
                sub.label(text=f"Baked res={obj['cumesh_resolution']} band={obj['cumesh_band']}")
                sub.label(text=f"      pb={obj['cumesh_project_back']:.2f} pad={obj['cumesh_scale_padding']:.2f}")
        else:
            layout.operator(CUMESH_OT_remesh.bl_idname, icon="MOD_REMESH")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
_classes = (
    CUMESH_AddonPreferences,
    CUMESH_Settings,
    CUMESH_OT_remesh,
    CUMESH_OT_refresh,
    CUMESH_OT_toggle_source,
    CUMESH_OT_select_source,
    CUMESH_PT_panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.cumesh_settings = PointerProperty(type=CUMESH_Settings)
    bpy.types.Object.cumesh_source = PointerProperty(
        type=bpy.types.Object,
        name="CuMesh Source",
        description="Original mesh this CuMesh result was generated from",
    )


def unregister():
    del bpy.types.Object.cumesh_source
    del bpy.types.Scene.cumesh_settings
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
