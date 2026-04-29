bl_info = {
    "name": "CuMesh Remesh",
    "author": "mikab",
    "version": (0, 4, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > CuMesh",
    "description": "Multi-backend remesh / repair: CuMesh, Manifold, PyMeshFix, PyMeshLab, Instant Meshes",
    "category": "Mesh",
}

import os
import sys
import tempfile
import subprocess
import shutil

import bpy
import bmesh
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, PointerProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup


# Threshold above which auto-stats after Remesh/Refresh is skipped to avoid
# stalling the UI on huge meshes (BFS over millions of verts is slow).
AUTO_STATS_MAX_VERTS = 500_000


WORKER_FILENAME = "cumesh_worker.py"


def _worker_path(name=WORKER_FILENAME):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------
class CUMESH_AddonPreferences(AddonPreferences):
    bl_idname = __name__

    python_exe: StringProperty(
        name="External Python",
        description="Path to a Python interpreter (venv) with torch + cumesh + (optional) manifold3d / pymeshfix / pymeshlab / pynanoinstantmeshes installed",
        subtype="FILE_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="External Python interpreter (venv with cumesh + optional libs):")
        col.prop(self, "python_exe")
        col.label(text="Example: C:/envs/cumesh/Scripts/python.exe", icon="INFO")
        col.separator()
        col.label(text="Optional libs (install in the same venv as needed):", icon="INFO")
        col.label(text="   pip install manifold3d pymeshfix pymeshlab pynanoinstantmeshes")
        col.separator()
        col.label(text=f"Workers folder: {os.path.dirname(_worker_path())}", icon="FILE_SCRIPT")


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


class MANIFOLD_Settings(PropertyGroup):
    refine: IntProperty(
        name="Refine",
        description="Edge subdivision count (0 = no refinement, just clean / make watertight)",
        default=0, min=0, max=8,
    )
    smooth: BoolProperty(
        name="Smooth by Normals",
        description="Apply Manifold's smooth_by_normals after construction",
        default=False,
    )


class MESHFIX_Settings(PropertyGroup):
    joincomp: BoolProperty(
        name="Join Components",
        description="Join all disconnected components into a single mesh",
        default=False,
    )
    remove_smallest: BoolProperty(
        name="Remove Smallest",
        description="Discard all components except the largest before repair",
        default=True,
    )


class PYMESHLAB_Settings(PropertyGroup):
    target_len: FloatProperty(
        name="Target Edge Length",
        description="Target triangle edge length in world units (0 = auto = 1% of bbox diag)",
        default=0.0, min=0.0, max=10.0, precision=4,
    )
    iterations: IntProperty(
        name="Iterations",
        description="Number of remeshing passes",
        default=3, min=1, max=20,
    )
    smooth_flag: BoolProperty(
        name="Smooth Pass",
        description="Apply per-iteration smoothing",
        default=True,
    )
    reproject_flag: BoolProperty(
        name="Reproject",
        description="Reproject vertices onto the original surface after each pass",
        default=True,
    )


class INSTANTMESHES_Settings(PropertyGroup):
    target_vertices: IntProperty(
        name="Target Vertices",
        description="Approximate number of vertices in the output mesh",
        default=10000, min=100, max=2_000_000,
    )
    output_quads: BoolProperty(
        name="Quads",
        description="Output quad mesh (rosy=4 posy=4) instead of tris (rosy=6 posy=6)",
        default=False,
    )
    smooth_iter: IntProperty(
        name="Smooth Iterations",
        description="Smoothing passes",
        default=2, min=0, max=10,
    )
    crease_angle: FloatProperty(
        name="Crease Angle",
        description="Dihedral angle threshold for hard edges (0 = disabled)",
        default=0.0, min=0.0, max=180.0,
    )
    align_boundaries: BoolProperty(
        name="Align Boundaries",
        description="Try to align result to mesh boundaries",
        default=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_python(prefs, worker_filename=WORKER_FILENAME):
    raw_py = (prefs.python_exe or "").strip()
    py_exe = bpy.path.abspath(raw_py) if raw_py else ""
    if not raw_py:
        return None, "'External Python' is empty. Edit > Preferences > Add-ons > CuMesh Remesh > set python.exe of your venv."
    if not os.path.isfile(py_exe):
        return None, f"External Python not found at: {py_exe}"
    worker = _worker_path(worker_filename)
    if not os.path.isfile(worker):
        return None, f"Worker script not found: {worker}"
    return (py_exe, worker), None


def _run_pipeline(context, src_obj, py_exe, worker, extra_args, label="worker"):
    """Run a worker subprocess on src_obj's evaluated mesh.

    `extra_args` is a list of extra CLI flags to pass to the worker (the
    --input/--output paths are added by this helper).

    Returns (mesh_datablock, error_message). On success error_message is None
    and mesh_datablock is a fresh bpy.data.meshes entry holding the result.
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

        cmd = [py_exe, worker, "--input", in_path, "--output", out_path] + list(extra_args)
        print(f"[{label}] Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
        except subprocess.TimeoutExpired:
            return None, f"{label} worker timed out (30 min)."

        if proc.stdout:
            print(f"[{label} stdout]\n" + proc.stdout)
        if proc.stderr:
            print(f"[{label} stderr]\n" + proc.stderr)

        if proc.returncode != 0:
            return None, f"{label} worker failed (code {proc.returncode}). See system console."
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


def _is_remesh_source(obj):
    """Any mesh is eligible as remesh source (allows chaining backends)."""
    return obj is not None and obj.type == "MESH"


def _build_cumesh_args(settings):
    args = ["--resolution", str(settings.resolution),
            "--band", str(settings.band),
            "--project-back", str(settings.project_back),
            "--scale-padding", str(settings.scale_padding)]
    if settings.verbose:
        args.append("--verbose")
    return args


def _build_manifold_args(settings, verbose):
    args = ["--refine", str(settings.refine)]
    if settings.smooth:
        args.append("--smooth")
    if verbose:
        args.append("--verbose")
    return args


def _build_meshfix_args(settings, verbose):
    args = []
    if settings.joincomp:
        args.append("--joincomp")
    if settings.remove_smallest:
        args.append("--remove-smallest")
    if verbose:
        args.append("--verbose")
    return args


def _build_pymeshlab_args(settings, verbose):
    args = ["--target-len", str(settings.target_len),
            "--iterations", str(settings.iterations)]
    if settings.smooth_flag:
        args.append("--smooth-flag")
    if settings.reproject_flag:
        args.append("--reproject-flag")
    if verbose:
        args.append("--verbose")
    return args


def _build_instantmeshes_args(settings, verbose):
    rosy = 4 if settings.output_quads else 6
    posy = 4 if settings.output_quads else 6
    args = ["--vertices", str(settings.target_vertices),
            "--rosy", str(rosy),
            "--posy", str(posy),
            "--smooth-iter", str(settings.smooth_iter),
            "--crease-angle", str(settings.crease_angle)]
    if settings.align_boundaries:
        args.append("--align-boundaries")
    if verbose:
        args.append("--verbose")
    return args


def _create_result_obj(context, src_obj, mesh, suffix, backend_name):
    """Wrap a fresh mesh datablock in a new object linked to src_obj."""
    mesh.name = src_obj.name + f"_{suffix}_mesh"
    new_obj = bpy.data.objects.new(src_obj.name + f"_{suffix}", mesh)
    context.collection.objects.link(new_obj)
    new_obj.matrix_world = src_obj.matrix_world.copy()
    new_obj.cumesh_source = src_obj
    new_obj["cumesh_backend"] = backend_name
    return new_obj


# ---------------------------------------------------------------------------
# Topology analysis
# ---------------------------------------------------------------------------
def _compute_topology_stats(mesh):
    """Return dict of {verts, faces, components, biggest_pct,
    non_manifold, boundary, zero_area_faces}. Cost ~O(verts+edges)."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    n_v = len(mesh.vertices)
    n_f = len(mesh.polygons)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bd = sum(1 for e in bm.edges if e.is_boundary)
    zf = sum(1 for f in bm.faces if f.calc_area() < 1e-12)

    visited = bytearray(n_v)
    biggest = 0
    comps = 0
    for vstart in bm.verts:
        if visited[vstart.index]:
            continue
        comps += 1
        size = 0
        stack = [vstart]
        while stack:
            v = stack.pop()
            if visited[v.index]:
                continue
            visited[v.index] = 1
            size += 1
            for e in v.link_edges:
                ov = e.other_vert(v)
                if not visited[ov.index]:
                    stack.append(ov)
        if size > biggest:
            biggest = size

    bm.free()
    return {
        "verts": n_v,
        "faces": n_f,
        "components": comps,
        "biggest_pct": (100.0 * biggest / n_v) if n_v else 0.0,
        "non_manifold": nm,
        "boundary": bd,
        "zero_area_faces": zf,
    }


def _store_stats(obj, stats):
    obj["cumesh_stat_verts"] = int(stats["verts"])
    obj["cumesh_stat_faces"] = int(stats["faces"])
    obj["cumesh_stat_comps"] = int(stats["components"])
    obj["cumesh_stat_biggest_pct"] = float(stats["biggest_pct"])
    obj["cumesh_stat_nonman"] = int(stats["non_manifold"])
    obj["cumesh_stat_boundary"] = int(stats["boundary"])
    obj["cumesh_stat_zero"] = int(stats["zero_area_faces"])


def _maybe_auto_stats(obj):
    """Compute and store stats only if mesh is small enough."""
    if obj is None or obj.type != "MESH":
        return
    if len(obj.data.vertices) <= AUTO_STATS_MAX_VERTS:
        try:
            _store_stats(obj, _compute_topology_stats(obj.data))
        except Exception as e:
            print(f"[CuMesh] auto-stats failed: {e}")


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
        return _is_remesh_source(context.active_object)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        settings = context.scene.cumesh_settings
        resolved, err = _resolve_python(prefs, "cumesh_worker.py")
        if err:
            self.report({"ERROR"}, "CuMesh: " + err); return {"CANCELLED"}
        py_exe, worker = resolved

        src_obj = context.active_object
        new_mesh, err = _run_pipeline(
            context, src_obj, py_exe, worker,
            _build_cumesh_args(settings), label="CuMesh",
        )
        if err:
            self.report({"ERROR"}, err); return {"CANCELLED"}

        new_obj = _create_result_obj(context, src_obj, new_mesh, "cumesh", "cumesh")
        _store_params(new_obj, settings)

        if settings.hide_source:
            src_obj.hide_set(True); src_obj.hide_render = True

        bpy.ops.object.select_all(action="DESELECT")
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        _maybe_auto_stats(new_obj)

        self.report({"INFO"}, f"CuMesh remesh done: {new_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_manifold(Operator):
    bl_idname = "cumesh.manifold_remesh"
    bl_label = "Manifold Remesh"
    bl_description = "Run Manifold (elalish/manifold) on the active mesh — requires near-watertight input (run PyMeshFix first if not)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _is_remesh_source(context.active_object)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        s = context.scene.cumesh_settings  # share verbose + hide_source
        ms = context.scene.manifold_settings
        resolved, err = _resolve_python(prefs, "manifold_worker.py")
        if err:
            self.report({"ERROR"}, "Manifold: " + err); return {"CANCELLED"}
        py_exe, worker = resolved

        src_obj = context.active_object

        # Pre-check: Manifold requires watertight input (no boundary edges,
        # no non-manifold edges). Catch this BEFORE spawning the subprocess
        # so the user gets an actionable message instead of "code 5".
        try:
            stats = _compute_topology_stats(src_obj.data)
            if stats["boundary"] > 0 or stats["non_manifold"] > 0:
                self.report(
                    {"ERROR"},
                    f"Manifold requires watertight input. "
                    f"'{src_obj.name}' has {stats['boundary']} boundary edges "
                    f"and {stats['non_manifold']} non-manifold edges. "
                    f"Run PyMeshFix Repair on this object first, then Manifold on the result.",
                )
                return {"CANCELLED"}
        except Exception:
            # If pre-check fails, fall through to the worker — it has its own checks.
            pass

        new_mesh, err = _run_pipeline(
            context, src_obj, py_exe, worker,
            _build_manifold_args(ms, s.verbose), label="Manifold",
        )
        if err:
            # Translate the worker's exit code 5 to something more actionable
            if "code 5" in err:
                err = (err + " Manifold requires a watertight input — run PyMeshFix Repair "
                              "on this object first.")
            self.report({"ERROR"}, err); return {"CANCELLED"}

        new_obj = _create_result_obj(context, src_obj, new_mesh, "manifold", "manifold")
        new_obj["manifold_refine"] = int(ms.refine)
        new_obj["manifold_smooth"] = bool(ms.smooth)

        if s.hide_source:
            src_obj.hide_set(True); src_obj.hide_render = True

        bpy.ops.object.select_all(action="DESELECT")
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        _maybe_auto_stats(new_obj)

        self.report({"INFO"}, f"Manifold done: {new_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_pymeshlab(Operator):
    bl_idname = "cumesh.pymeshlab_remesh"
    bl_label = "PyMeshLab Isotropic"
    bl_description = "Run PyMeshLab's isotropic explicit remesher (clean equilateral triangles, GPL)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _is_remesh_source(context.active_object)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        s = context.scene.cumesh_settings
        ps = context.scene.pymeshlab_settings
        resolved, err = _resolve_python(prefs, "pymeshlab_worker.py")
        if err:
            self.report({"ERROR"}, "PyMeshLab: " + err); return {"CANCELLED"}
        py_exe, worker = resolved

        src_obj = context.active_object
        new_mesh, err = _run_pipeline(
            context, src_obj, py_exe, worker,
            _build_pymeshlab_args(ps, s.verbose), label="PyMeshLab",
        )
        if err:
            self.report({"ERROR"}, err); return {"CANCELLED"}

        new_obj = _create_result_obj(context, src_obj, new_mesh, "pymeshlab", "pymeshlab")
        new_obj["pymeshlab_target_len"] = float(ps.target_len)
        new_obj["pymeshlab_iterations"] = int(ps.iterations)

        if s.hide_source:
            src_obj.hide_set(True); src_obj.hide_render = True

        bpy.ops.object.select_all(action="DESELECT")
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        _maybe_auto_stats(new_obj)

        self.report({"INFO"}, f"PyMeshLab done: {new_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_instantmeshes(Operator):
    bl_idname = "cumesh.instantmeshes_remesh"
    bl_label = "Instant Meshes"
    bl_description = "Run Wenzel Jakob's Instant Meshes via pynanoinstantmeshes (field-aligned remesher, GPL)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _is_remesh_source(context.active_object)

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        s = context.scene.cumesh_settings
        ims = context.scene.instantmeshes_settings

        resolved, err = _resolve_python(prefs, "instant_meshes_worker.py")
        if err:
            self.report({"ERROR"}, "Instant Meshes: " + err); return {"CANCELLED"}
        py_exe, worker = resolved

        src_obj = context.active_object
        new_mesh, err = _run_pipeline(
            context, src_obj, py_exe, worker,
            _build_instantmeshes_args(ims, s.verbose), label="InstantMeshes",
        )
        if err:
            # 0xC0000409 = STATUS_STACK_BUFFER_OVERRUN, common pynanoinstantmeshes
            # crash on dense/noisy inputs — translate to friendly hint
            if "3221226505" in err or "0xC0000409" in err.upper():
                err = (err + " [pynanoinstantmeshes C++ crashed during position-field optimization. "
                              "Try: lower 'Target Vertices', pre-clean with PyMeshFix, "
                              "or use PyMeshLab Isotropic instead.]")
            self.report({"ERROR"}, err); return {"CANCELLED"}

        new_obj = _create_result_obj(context, src_obj, new_mesh, "instantmeshes", "instantmeshes")
        new_obj["instantmeshes_target_vertices"] = int(ims.target_vertices)
        new_obj["instantmeshes_quads"] = bool(ims.output_quads)

        if s.hide_source:
            src_obj.hide_set(True); src_obj.hide_render = True

        bpy.ops.object.select_all(action="DESELECT")
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        _maybe_auto_stats(new_obj)

        self.report({"INFO"}, f"Instant Meshes done: {new_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_meshfix(Operator):
    bl_idname = "cumesh.meshfix_repair"
    bl_label = "PyMeshFix Repair"
    bl_description = "Run PyMeshFix (Marco Attene's MeshFix) on the active mesh — fills holes, removes self-intersections, makes manifold"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        s = context.scene.cumesh_settings
        fs = context.scene.meshfix_settings
        resolved, err = _resolve_python(prefs, "meshfix_worker.py")
        if err:
            self.report({"ERROR"}, "MeshFix: " + err); return {"CANCELLED"}
        py_exe, worker = resolved

        src_obj = context.active_object
        new_mesh, err = _run_pipeline(
            context, src_obj, py_exe, worker,
            _build_meshfix_args(fs, s.verbose), label="MeshFix",
        )
        if err:
            self.report({"ERROR"}, err); return {"CANCELLED"}

        new_obj = _create_result_obj(context, src_obj, new_mesh, "meshfix", "meshfix")
        new_obj["meshfix_joincomp"] = bool(fs.joincomp)
        new_obj["meshfix_remove_smallest"] = bool(fs.remove_smallest)

        if s.hide_source:
            src_obj.hide_set(True); src_obj.hide_render = True

        bpy.ops.object.select_all(action="DESELECT")
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        _maybe_auto_stats(new_obj)

        self.report({"INFO"}, f"MeshFix done: {new_obj.name}")
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

        result_obj = context.active_object
        src_obj = result_obj.cumesh_source
        if src_obj is None or src_obj.type != "MESH":
            self.report({"ERROR"}, "Linked source is missing or not a mesh.")
            return {"CANCELLED"}

        backend = result_obj.get("cumesh_backend", "cumesh")
        worker_map = {
            "cumesh": ("cumesh_worker.py", _build_cumesh_args(settings), "CuMesh"),
            "manifold": ("manifold_worker.py",
                         _build_manifold_args(context.scene.manifold_settings, settings.verbose),
                         "Manifold"),
            "meshfix": ("meshfix_worker.py",
                        _build_meshfix_args(context.scene.meshfix_settings, settings.verbose),
                        "MeshFix"),
            "pymeshlab": ("pymeshlab_worker.py",
                          _build_pymeshlab_args(context.scene.pymeshlab_settings, settings.verbose),
                          "PyMeshLab"),
            "instantmeshes": ("instant_meshes_worker.py",
                              _build_instantmeshes_args(
                                  context.scene.instantmeshes_settings,
                                  settings.verbose),
                              "InstantMeshes"),
        }
        if backend not in worker_map:
            self.report({"ERROR"}, f"Unknown backend '{backend}'."); return {"CANCELLED"}
        worker_file, extra_args, label = worker_map[backend]

        resolved, err = _resolve_python(prefs, worker_file)
        if err:
            self.report({"ERROR"}, f"{label}: " + err); return {"CANCELLED"}
        py_exe, worker = resolved

        new_mesh, err = _run_pipeline(context, src_obj, py_exe, worker, extra_args, label=label)
        if err:
            self.report({"ERROR"}, err); return {"CANCELLED"}

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
        if backend == "cumesh":
            _store_params(result_obj, settings)
        _maybe_auto_stats(result_obj)

        self.report({"INFO"}, f"{label} refreshed: {result_obj.name}")
        return {"FINISHED"}


class CUMESH_OT_analyse(Operator):
    bl_idname = "cumesh.analyse"
    bl_label = "Analyse Topology"
    bl_description = "Compute topology stats (components, non-manifold, boundary, zero-area faces) for the active mesh"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        try:
            stats = _compute_topology_stats(obj.data)
        except Exception as e:
            self.report({"ERROR"}, f"Analyse failed: {e}")
            return {"CANCELLED"}
        _store_stats(obj, stats)
        self.report(
            {"INFO"},
            f"{obj.name}: v={stats['verts']} f={stats['faces']} "
            f"comps={stats['components']} (biggest {stats['biggest_pct']:.1f}%) "
            f"nonman={stats['non_manifold']} boundary={stats['boundary']} "
            f"zero={stats['zero_area_faces']}",
        )
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
        ms = context.scene.manifold_settings
        fs = context.scene.meshfix_settings
        ps = context.scene.pymeshlab_settings
        ims = context.scene.instantmeshes_settings
        obj = context.active_object

        layout.prop(s, "hide_source")
        layout.prop(s, "verbose")

        # ---- Backend: CuMesh ----
        box = layout.box()
        box.label(text="CuMesh (GPU narrow-band DC)", icon="MOD_REMESH")
        col = box.column(align=True)
        col.prop(s, "resolution")
        col.prop(s, "band")
        col.prop(s, "project_back", slider=True)
        col.prop(s, "scale_padding")
        box.operator(CUMESH_OT_remesh.bl_idname, icon="MOD_REMESH")

        # ---- Backend: Manifold ----
        box = layout.box()
        box.label(text="Manifold (CPU CSG / refine)", icon="MOD_BOOLEAN")
        col = box.column(align=True)
        col.prop(ms, "refine")
        col.prop(ms, "smooth")
        box.operator(CUMESH_OT_manifold.bl_idname, icon="MOD_BOOLEAN")

        # ---- Backend: PyMeshFix ----
        box = layout.box()
        box.label(text="PyMeshFix (repair)", icon="TOOL_SETTINGS")
        col = box.column(align=True)
        col.prop(fs, "joincomp")
        col.prop(fs, "remove_smallest")
        box.operator(CUMESH_OT_meshfix.bl_idname, icon="TOOL_SETTINGS")

        # ---- Backend: PyMeshLab ----
        box = layout.box()
        box.label(text="PyMeshLab (isotropic remesh)", icon="MESH_GRID")
        col = box.column(align=True)
        col.prop(ps, "target_len")
        col.prop(ps, "iterations")
        row = box.row(align=True)
        row.prop(ps, "smooth_flag")
        row.prop(ps, "reproject_flag")
        box.operator(CUMESH_OT_pymeshlab.bl_idname, icon="MESH_GRID")

        # ---- Backend: Instant Meshes ----
        box = layout.box()
        box.label(text="Instant Meshes (field-aligned)", icon="MESH_ICOSPHERE")
        col = box.column(align=True)
        col.prop(ims, "target_vertices")
        col.prop(ims, "output_quads")
        col.prop(ims, "smooth_iter")
        col.prop(ims, "crease_angle")
        col.prop(ims, "align_boundaries")
        box.operator(CUMESH_OT_instantmeshes.bl_idname, icon="MESH_ICOSPHERE")

        # ---- Linked-result section ----
        if _is_cumesh_result(obj):
            layout.separator()
            box = layout.box()
            backend = obj.get("cumesh_backend", "cumesh")
            box.label(text=f"Linked Result — backend: {backend}", icon="LINKED")
            src = obj.cumesh_source
            row = box.row()
            row.label(text=f"Source: {src.name if src else '<missing>'}", icon="OBJECT_DATA")
            box.operator(CUMESH_OT_refresh.bl_idname, icon="FILE_REFRESH")
            row = box.row(align=True)
            row.operator(CUMESH_OT_select_source.bl_idname, icon="EDITMODE_HLT")
            row.operator(CUMESH_OT_toggle_source.bl_idname, icon="HIDE_OFF")

            sub = box.column(align=True)
            sub.scale_y = 0.85
            if backend == "cumesh" and "cumesh_resolution" in obj:
                sub.label(text=f"Baked res={obj['cumesh_resolution']} band={obj['cumesh_band']}")
                sub.label(text=f"      pb={obj['cumesh_project_back']:.2f} pad={obj['cumesh_scale_padding']:.2f}")
            elif backend == "manifold":
                sub.label(text=f"Baked refine={obj.get('manifold_refine','-')} smooth={obj.get('manifold_smooth','-')}")
            elif backend == "pymeshlab":
                sub.label(text=f"Baked target_len={obj.get('pymeshlab_target_len','-')} iters={obj.get('pymeshlab_iterations','-')}")
            elif backend == "instantmeshes":
                sub.label(text=f"Baked verts={obj.get('instantmeshes_target_vertices','-')} quads={obj.get('instantmeshes_quads','-')}")
            elif backend == "meshfix":
                sub.label(text=f"Baked join={obj.get('meshfix_joincomp','-')} kill_small={obj.get('meshfix_remove_smallest','-')}")

        # Topology stats — works on any active mesh, not just cumesh results
        if obj is not None and obj.type == "MESH":
            layout.separator()
            box = layout.box()
            box.label(text="Topology Stats", icon="MESH_DATA")
            box.operator(CUMESH_OT_analyse.bl_idname, icon="VIEWZOOM")
            if "cumesh_stat_verts" in obj:
                col = box.column(align=True)
                col.scale_y = 0.85
                col.label(text=f"Verts: {obj['cumesh_stat_verts']:,}".replace(",", " "))
                col.label(text=f"Faces: {obj['cumesh_stat_faces']:,}".replace(",", " "))
                col.label(text=f"Components: {obj['cumesh_stat_comps']}  (biggest {obj['cumesh_stat_biggest_pct']:.1f}%)")
                col.label(text=f"Non-manifold edges: {obj['cumesh_stat_nonman']}")
                col.label(text=f"Boundary edges: {obj['cumesh_stat_boundary']}")
                col.label(text=f"Zero-area faces: {obj['cumesh_stat_zero']}")
            else:
                col = box.column()
                col.scale_y = 0.85
                col.label(text="(click Analyse Topology to compute)", icon="INFO")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
_classes = (
    CUMESH_AddonPreferences,
    CUMESH_Settings,
    MANIFOLD_Settings,
    MESHFIX_Settings,
    PYMESHLAB_Settings,
    INSTANTMESHES_Settings,
    CUMESH_OT_remesh,
    CUMESH_OT_manifold,
    CUMESH_OT_pymeshlab,
    CUMESH_OT_instantmeshes,
    CUMESH_OT_meshfix,
    CUMESH_OT_refresh,
    CUMESH_OT_analyse,
    CUMESH_OT_toggle_source,
    CUMESH_OT_select_source,
    CUMESH_PT_panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.cumesh_settings = PointerProperty(type=CUMESH_Settings)
    bpy.types.Scene.manifold_settings = PointerProperty(type=MANIFOLD_Settings)
    bpy.types.Scene.meshfix_settings = PointerProperty(type=MESHFIX_Settings)
    bpy.types.Scene.pymeshlab_settings = PointerProperty(type=PYMESHLAB_Settings)
    bpy.types.Scene.instantmeshes_settings = PointerProperty(type=INSTANTMESHES_Settings)
    bpy.types.Object.cumesh_source = PointerProperty(
        type=bpy.types.Object,
        name="Remesh Source",
        description="Original mesh this remesh result was generated from",
    )


def unregister():
    del bpy.types.Object.cumesh_source
    del bpy.types.Scene.instantmeshes_settings
    del bpy.types.Scene.pymeshlab_settings
    del bpy.types.Scene.meshfix_settings
    del bpy.types.Scene.manifold_settings
    del bpy.types.Scene.cumesh_settings
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
