bl_info = {
    "name": "CuMesh Remesh",
    "author": "bebelllm",
    "version": (0, 1, 0),
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
# Preferences: path to the external Python interpreter that has torch+cumesh
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
# Operator
# ---------------------------------------------------------------------------
class CUMESH_OT_remesh(Operator):
    bl_idname = "cumesh.remesh"
    bl_label = "CuMesh Remesh"
    bl_description = "Remesh the active mesh using CuMesh narrow-band dual contouring"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        settings = context.scene.cumesh_settings

        raw_py = (prefs.python_exe or "").strip()
        py_exe = bpy.path.abspath(raw_py) if raw_py else ""
        if not raw_py:
            self.report(
                {"ERROR"},
                "CuMesh: 'External Python' is empty. Edit > Preferences > Add-ons > CuMesh Remesh > set python.exe of your torch+cumesh venv.",
            )
            return {"CANCELLED"}
        if not os.path.isfile(py_exe):
            self.report(
                {"ERROR"},
                f"CuMesh: External Python not found at: {py_exe}",
            )
            return {"CANCELLED"}

        worker = _worker_path()
        if not os.path.isfile(worker):
            self.report({"ERROR"}, f"Worker script not found: {worker}")
            return {"CANCELLED"}

        src_obj = context.active_object
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = src_obj.evaluated_get(depsgraph)

        tmpdir = tempfile.mkdtemp(prefix="cumesh_")
        in_path = os.path.join(tmpdir, "input.ply")
        out_path = os.path.join(tmpdir, "output.ply")

        try:
            # Export the evaluated mesh (modifiers applied) to PLY.
            # Use a temporary selection so ply_export writes only this object.
            prev_active = context.view_layer.objects.active
            prev_selected = [o for o in context.selected_objects]
            bpy.ops.object.select_all(action="DESELECT")
            # Put the evaluated result into a temp mesh object so export is clean.
            tmp_mesh = bpy.data.meshes.new_from_object(eval_obj)
            tmp_obj = bpy.data.objects.new("_cumesh_tmp_in", tmp_mesh)
            context.collection.objects.link(tmp_obj)
            tmp_obj.matrix_world = src_obj.matrix_world.copy()
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

            # Cleanup temp export object
            bpy.data.objects.remove(tmp_obj, do_unlink=True)
            bpy.data.meshes.remove(tmp_mesh, do_unlink=True)

            # Restore selection
            for o in prev_selected:
                try:
                    o.select_set(True)
                except ReferenceError:
                    pass
            if prev_active:
                context.view_layer.objects.active = prev_active

            # Build command
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
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60 * 30,
                )
            except subprocess.TimeoutExpired:
                self.report({"ERROR"}, "CuMesh worker timed out (30 min).")
                return {"CANCELLED"}

            if proc.stdout:
                print("[CuMesh stdout]\n" + proc.stdout)
            if proc.stderr:
                print("[CuMesh stderr]\n" + proc.stderr)

            if proc.returncode != 0:
                self.report({"ERROR"}, f"CuMesh worker failed (code {proc.returncode}). See system console.")
                return {"CANCELLED"}

            if not os.path.isfile(out_path):
                self.report({"ERROR"}, "Worker finished but no output file was produced.")
                return {"CANCELLED"}

            # Import result
            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.wm.ply_import(
                filepath=out_path,
                forward_axis="Y",
                up_axis="Z",
            )
            new_obj = context.view_layer.objects.active
            if new_obj is None:
                self.report({"ERROR"}, "PLY import did not create an object.")
                return {"CANCELLED"}

            new_obj.name = src_obj.name + "_cumesh"
            new_obj.matrix_world = src_obj.matrix_world.copy()

            if settings.hide_source:
                src_obj.hide_set(True)
                src_obj.hide_render = True

            # Re-select new object
            bpy.ops.object.select_all(action="DESELECT")
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj

            self.report({"INFO"}, f"CuMesh remesh done: {new_obj.name}")
            return {"FINISHED"}

        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


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

        col = layout.column(align=True)
        col.prop(s, "resolution")
        col.prop(s, "band")
        col.prop(s, "project_back", slider=True)
        col.prop(s, "scale_padding")

        layout.separator()
        layout.prop(s, "hide_source")
        layout.prop(s, "verbose")

        layout.separator()
        layout.operator(CUMESH_OT_remesh.bl_idname, icon="MOD_REMESH")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
_classes = (
    CUMESH_AddonPreferences,
    CUMESH_Settings,
    CUMESH_OT_remesh,
    CUMESH_PT_panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.cumesh_settings = PointerProperty(type=CUMESH_Settings)


def unregister():
    del bpy.types.Scene.cumesh_settings
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
