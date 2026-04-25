"""Worker for the Instant Meshes backend.

Wraps Wenzel Jakob's Instant Meshes standalone binary
(https://github.com/wjakob/instant-meshes/releases). Field-aligned quad/tri
remesher with explicit target vertex count.

Setup:
  1. Download the binary release for your OS:
     https://github.com/wjakob/instant-meshes/releases
  2. Set its full path in the addon preferences ("Instant Meshes Exe").

The binary path comes through --exe (resolved by the addon from prefs).
"""
import argparse
import os
import struct
import subprocess
import sys


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
        if f.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        if f.readline().strip().decode("ascii") != "format binary_little_endian 1.0":
            raise ValueError("Unsupported PLY format")
        elements = []; current = None
        while True:
            raw = f.readline()
            if not raw: raise ValueError("EOF in header")
            line = raw.strip().decode("ascii")
            if line == "end_header": break
            if line.startswith("comment") or line.startswith("obj_info"): continue
            parts = line.split()
            if parts[0] == "element":
                if current is not None: elements.append(current)
                current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elif parts[0] == "property":
                if parts[1] == "list":
                    current["props"].append({"list": True, "count_type": parts[2], "elem_type": parts[3], "name": parts[4]})
                else:
                    current["props"].append({"list": False, "type": parts[1], "name": parts[2]})
        if current is not None: elements.append(current)
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
                if _PLY_TYPE_TO_NP[p["elem_type"]][0].startswith("u"):
                    elem_fmt_char = elem_fmt_char.upper()
                faces_list = []
                for _ in range(elem["count"]):
                    n = struct.unpack(count_fmt, f.read(count_size))[0]
                    idx = struct.unpack("<" + elem_fmt_char * n, f.read(elem_size * n))
                    if n == 3: faces_list.append(idx)
                    elif n == 4:
                        # quad → fan triangulate
                        faces_list.append((idx[0], idx[1], idx[2]))
                        faces_list.append((idx[0], idx[2], idx[3]))
                    elif n > 4:
                        for i in range(1, n - 1): faces_list.append((idx[0], idx[i], idx[i + 1]))
                faces = np.asarray(faces_list, dtype=np.int32)
            else:
                rec_size = sum(_PLY_TYPE_TO_NP[p["type"]][1] for p in elem["props"])
                f.read(rec_size * elem["count"])
    if vertices is None or faces is None:
        raise ValueError("PLY missing vertex/face element")
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
    face_rec["n"] = 3; face_rec["v"] = faces
    with open(path, "wb") as f:
        f.write(header); f.write(vertices.tobytes(order="C")); f.write(face_rec.tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser(description="Instant Meshes wrapper worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exe", required=True,
                        help="Full path to the Instant Meshes binary")
    parser.add_argument("--vertices", type=int, default=10000,
                        help="Target output vertex count")
    parser.add_argument("--rosy", type=int, default=4, choices=[2, 4, 6],
                        help="Orientation symmetry (2/4/6); 4 = quads, 6 = tris")
    parser.add_argument("--posy", type=int, default=4, choices=[4, 6],
                        help="Position symmetry (4 = quads, 6 = tris)")
    parser.add_argument("--smooth", type=int, default=2,
                        help="Smoothing iterations")
    parser.add_argument("--crease", type=float, default=0.0,
                        help="Crease angle in degrees (0 = none)")
    parser.add_argument("--align-boundaries", action="store_true",
                        help="Align result to mesh boundaries")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.exe):
        print(f"ERROR: Instant Meshes binary not found at: {args.exe}\n"
              "Download from: https://github.com/wjakob/instant-meshes/releases\n"
              "Then set its path in the addon preferences.", file=sys.stderr)
        sys.exit(2)

    cmd = [
        args.exe,
        args.input,
        "--output", args.output,
        "--vertices", str(args.vertices),
        "--rosy", str(args.rosy),
        "--posy", str(args.posy),
        "--smooth", str(args.smooth),
        "--deterministic",
    ]
    if args.crease > 0:
        cmd += ["--crease", str(args.crease)]
    if args.align_boundaries:
        cmd.append("--align-to-boundaries")

    if args.verbose:
        print(f"[instant_meshes] running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
    except subprocess.TimeoutExpired:
        print("ERROR: Instant Meshes timed out (30 min).", file=sys.stderr); sys.exit(7)
    if args.verbose:
        if proc.stdout: print(proc.stdout)
        if proc.stderr: print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print(f"ERROR: Instant Meshes exited with code {proc.returncode}\n"
              f"stderr: {proc.stderr[-2000:]}", file=sys.stderr)
        sys.exit(proc.returncode)
    if not os.path.isfile(args.output):
        print(f"ERROR: Instant Meshes did not produce {args.output}", file=sys.stderr)
        sys.exit(8)

    # Re-write the PLY through our writer to ensure binary_little_endian + tri
    # layout that the Blender addon expects (Instant Meshes may emit quads).
    try:
        v, f = _read_ply(args.output)
    except Exception as e:
        print(f"ERROR: failed to re-read Instant Meshes PLY output: {e}", file=sys.stderr)
        sys.exit(9)
    if args.verbose:
        print(f"[instant_meshes] output: {v.shape[0]} verts, {f.shape[0]} tris (after triangulation)")
    try:
        _write_ply(args.output, v, f)
    except Exception as e:
        print(f"ERROR: failed to rewrite output: {e}", file=sys.stderr); sys.exit(10)


if __name__ == "__main__":
    main()
