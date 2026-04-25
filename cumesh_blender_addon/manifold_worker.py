"""Worker for the Manifold backend (https://github.com/elalish/manifold).

Wraps `manifold3d` Python bindings. Manifold REQUIRES watertight input;
if the input is non-manifold the constructor returns an empty Manifold.
We attempt a vertex merge before constructing as a best-effort fix.

Install in the venv: pip install manifold3d
"""
import argparse
import struct
import sys


# ---------------------------------------------------------------------------
# Minimal binary-little-endian PLY I/O (numpy only) — copy of cumesh_worker
# ---------------------------------------------------------------------------
_PLY_TYPE_TO_NP = {
    "char": ("i1", 1), "int8": ("i1", 1),
    "uchar": ("u1", 1), "uint8": ("u1", 1),
    "short": ("i2", 2), "int16": ("i2", 2),
    "ushort": ("u2", 2), "uint16": ("u2", 2),
    "int": ("i4", 4), "int32": ("i4", 4),
    "uint": ("u4", 4), "uint32": ("u4", 4),
    "float": ("f4", 4), "float32": ("f4", 4),
    "double": ("f8", 8), "float64": ("f8", 8),
}


def _read_ply(path):
    import numpy as np
    with open(path, "rb") as f:
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt_line = f.readline().strip().decode("ascii")
        if fmt_line != "format binary_little_endian 1.0":
            raise ValueError(f"Unsupported PLY format: {fmt_line}")
        elements = []
        current = None
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("EOF in PLY header")
            line = raw.strip().decode("ascii")
            if line == "end_header":
                break
            if line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "element":
                if current is not None:
                    elements.append(current)
                current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elif parts[0] == "property":
                if parts[1] == "list":
                    current["props"].append({"list": True, "count_type": parts[2], "elem_type": parts[3], "name": parts[4]})
                else:
                    current["props"].append({"list": False, "type": parts[1], "name": parts[2]})
        if current is not None:
            elements.append(current)
        vertices = None; faces = None
        for elem in elements:
            if elem["name"] == "vertex":
                dt = np.dtype([(p["name"], "<" + _PLY_TYPE_TO_NP[p["type"]][0]) for p in elem["props"] if not p["list"]])
                data = np.frombuffer(f.read(dt.itemsize * elem["count"]), dtype=dt)
                vertices = np.stack([data["x"].astype(np.float32), data["y"].astype(np.float32), data["z"].astype(np.float32)], axis=1)
            elif elem["name"] == "face":
                p = elem["props"][0]
                count_size = _PLY_TYPE_TO_NP[p["count_type"]][1]
                elem_size = _PLY_TYPE_TO_NP[p["elem_type"]][1]
                count_fmt = "<" + {1: "B", 2: "H", 4: "I"}[count_size]
                elem_fmt_char = {1: "i", 2: "i", 4: "i"}[elem_size]
                elem_fmt_char = elem_fmt_char.upper() if _PLY_TYPE_TO_NP[p["elem_type"]][0].startswith("u") else elem_fmt_char
                faces_list = []
                for _ in range(elem["count"]):
                    n = struct.unpack(count_fmt, f.read(count_size))[0]
                    idx = struct.unpack("<" + elem_fmt_char * n, f.read(elem_size * n))
                    if n == 3:
                        faces_list.append(idx)
                    elif n > 3:
                        for i in range(1, n - 1):
                            faces_list.append((idx[0], idx[i], idx[i + 1]))
                faces = np.asarray(faces_list, dtype=np.int32)
            else:
                rec_size = sum(_PLY_TYPE_TO_NP[p["type"]][1] for p in elem["props"])
                f.read(rec_size * elem["count"])
    if vertices is None or faces is None:
        raise ValueError("PLY missing vertex or face element")
    return vertices, faces


def _write_ply(path, vertices, faces):
    import numpy as np
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar uint vertex_indices\nend_header\n"
    ).encode("ascii")
    face_rec = np.empty(faces.shape[0], dtype=np.dtype([("n", "u1"), ("v", "<u4", 3)]))
    face_rec["n"] = 3
    face_rec["v"] = faces
    with open(path, "wb") as f:
        f.write(header); f.write(vertices.tobytes(order="C")); f.write(face_rec.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Manifold remesh / repair worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--refine", type=int, default=0,
                        help="Edge subdivision count (0=no refinement)")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply normal-based smoothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as e:
        print(f"ERROR: numpy not importable: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        import manifold3d as m3
    except ImportError as e:
        print(f"ERROR: manifold3d not importable: {e}\n"
              "Install in this venv with: pip install manifold3d", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print(f"[manifold] manifold3d ok, loading {args.input}")

    try:
        np_v, np_f = _read_ply(args.input)
    except Exception as e:
        print(f"ERROR: failed to read input PLY: {e}", file=sys.stderr); sys.exit(4)
    if np_v.shape[0] == 0 or np_f.shape[0] == 0:
        print("ERROR: empty input mesh", file=sys.stderr); sys.exit(4)

    if args.verbose:
        print(f"[manifold] input: {np_v.shape[0]} verts, {np_f.shape[0]} faces")

    mesh_in = m3.Mesh(
        vert_properties=np_v.astype(np.float32),
        tri_verts=np_f.astype(np.uint32),
    )
    # Try to merge near-duplicate vertices first (helps non-watertight inputs)
    try:
        mesh_in.merge()
    except Exception:
        pass

    manifold = m3.Manifold(mesh_in)
    status = manifold.status()
    if args.verbose:
        print(f"[manifold] status after construction: {status}")

    if manifold.is_empty():
        print(f"ERROR: Manifold construction returned empty (status={status}).\n"
              "Manifold requires a watertight input. Try repairing the mesh first "
              "(e.g. PyMeshFix or Blender's 'Make Manifold').", file=sys.stderr)
        sys.exit(5)

    if args.refine > 0:
        if args.verbose: print(f"[manifold] refine x{args.refine}")
        manifold = manifold.refine(args.refine)
    if args.smooth:
        if args.verbose: print("[manifold] smooth_by_normals")
        try:
            manifold = manifold.smooth_by_normals(0)
        except Exception as e:
            print(f"WARNING: smooth_by_normals failed: {e}", file=sys.stderr)

    out_mesh = manifold.to_mesh()
    out_v = np.asarray(out_mesh.vert_properties[:, :3], dtype=np.float32)
    out_f = np.asarray(out_mesh.tri_verts, dtype=np.int32)

    if args.verbose:
        print(f"[manifold] output: {out_v.shape[0]} verts, {out_f.shape[0]} faces")

    try:
        _write_ply(args.output, out_v, out_f)
    except Exception as e:
        print(f"ERROR: failed to write output PLY: {e}", file=sys.stderr); sys.exit(6)


if __name__ == "__main__":
    main()
