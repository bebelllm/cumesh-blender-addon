"""
Standalone worker script invoked by the Blender CuMesh addon.

Runs in an *external* Python interpreter (a venv) that has:
    - torch (CUDA build, >= 2.4)
    - cumesh

No other third-party dependency is required: PLY I/O is handled with numpy
only (numpy ships with torch).

Reads an input PLY mesh, runs `cumesh.remeshing.remesh_narrow_band_dc`,
and writes the result to an output PLY.
"""
import argparse
import struct
import sys


# ---------------------------------------------------------------------------
# Minimal binary-little-endian PLY reader / writer (numpy only).
#
# The addon exports the input PLY with only x,y,z vertex properties and
# triangle faces (no UVs, no normals, no colors), so we only need to handle
# that specific layout on the read side. On the write side we emit the same
# minimal layout.
# ---------------------------------------------------------------------------
_PLY_TYPE_TO_NP = {
    "char": ("i1", 1),
    "int8": ("i1", 1),
    "uchar": ("u1", 1),
    "uint8": ("u1", 1),
    "short": ("i2", 2),
    "int16": ("i2", 2),
    "ushort": ("u2", 2),
    "uint16": ("u2", 2),
    "int": ("i4", 4),
    "int32": ("i4", 4),
    "uint": ("u4", 4),
    "uint32": ("u4", 4),
    "float": ("f4", 4),
    "float32": ("f4", 4),
    "double": ("f8", 8),
    "float64": ("f8", 8),
}


def _read_ply(path):
    """Read a binary-little-endian PLY file and return (vertices, faces)
    as numpy arrays. vertices is (N,3) float32, faces is (M,3) int32."""
    import numpy as np

    with open(path, "rb") as f:
        # --- header ---
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt_line = f.readline().strip().decode("ascii")
        if fmt_line != "format binary_little_endian 1.0":
            raise ValueError(
                f"Unsupported PLY format (expected binary_little_endian 1.0): {fmt_line}"
            )

        elements = []  # list of (name, count, properties)
        current = None
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("Unexpected EOF in PLY header")
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
                if current is None:
                    raise ValueError(f"property before element: {line}")
                if parts[1] == "list":
                    # property list <count_type> <elem_type> <name>
                    current["props"].append(
                        {"list": True, "count_type": parts[2], "elem_type": parts[3], "name": parts[4]}
                    )
                else:
                    current["props"].append({"list": False, "type": parts[1], "name": parts[2]})
            else:
                # ignore unknown header lines
                pass
        if current is not None:
            elements.append(current)

        # --- body ---
        vertices = None
        faces = None

        for elem in elements:
            if elem["name"] == "vertex":
                # Build a structured numpy dtype describing one vertex record
                dtype_fields = []
                for p in elem["props"]:
                    if p["list"]:
                        raise ValueError("List property on vertex element not supported")
                    np_t, _ = _PLY_TYPE_TO_NP[p["type"]]
                    dtype_fields.append((p["name"], "<" + np_t))
                dt = np.dtype(dtype_fields)
                data = np.frombuffer(f.read(dt.itemsize * elem["count"]), dtype=dt)
                vertices = np.stack(
                    [data["x"].astype(np.float32),
                     data["y"].astype(np.float32),
                     data["z"].astype(np.float32)],
                    axis=1,
                )
            elif elem["name"] == "face":
                # Faces are a list property: read them one by one because the
                # list length can vary (we still expect 3 here).
                if len(elem["props"]) != 1 or not elem["props"][0]["list"]:
                    raise ValueError("Unsupported face element layout")
                p = elem["props"][0]
                count_np_t, count_size = _PLY_TYPE_TO_NP[p["count_type"]]
                elem_np_t, elem_size = _PLY_TYPE_TO_NP[p["elem_type"]]
                count_fmt = "<" + {1: "b", 2: "h", 4: "i"}[count_size].replace(
                    "b", "B" if count_np_t.startswith("u") else "b"
                ).replace(
                    "h", "H" if count_np_t.startswith("u") else "h"
                ).replace(
                    "i", "I" if count_np_t.startswith("u") else "i"
                )
                elem_fmt_char = {1: "b", 2: "h", 4: "i"}[elem_size]
                if elem_np_t.startswith("u"):
                    elem_fmt_char = elem_fmt_char.upper()
                faces_list = []
                for _ in range(elem["count"]):
                    n = struct.unpack(count_fmt, f.read(count_size))[0]
                    idx = struct.unpack("<" + elem_fmt_char * n, f.read(elem_size * n))
                    if n == 3:
                        faces_list.append(idx)
                    elif n > 3:
                        # fan-triangulate, just in case
                        for i in range(1, n - 1):
                            faces_list.append((idx[0], idx[i], idx[i + 1]))
                    # n<3 -> skip
                faces = np.asarray(faces_list, dtype=np.int32)
            else:
                # Skip any other element by computing its per-record size.
                for p in elem["props"]:
                    if p["list"]:
                        raise ValueError(
                            f"Cannot skip element '{elem['name']}' with list property"
                        )
                rec_size = sum(_PLY_TYPE_TO_NP[p["type"]][1] for p in elem["props"])
                f.read(rec_size * elem["count"])

    if vertices is None or faces is None:
        raise ValueError("PLY is missing vertex or face element")
    return vertices, faces


def _write_ply(path, vertices, faces):
    """Write a minimal binary-little-endian PLY with x,y,z floats and
    uchar+uint triangle face lists."""
    import numpy as np

    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    # Build face records: for each face, 1 byte count=3 + 3 * uint32
    face_rec = np.empty(
        faces.shape[0],
        dtype=np.dtype([("n", "u1"), ("v", "<u4", 3)]),
    )
    face_rec["n"] = 3
    face_rec["v"] = faces

    with open(path, "wb") as f:
        f.write(header)
        f.write(vertices.tobytes(order="C"))
        f.write(face_rec.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CuMesh narrow-band DC remesh worker")
    parser.add_argument("--input", required=True, help="Input PLY path")
    parser.add_argument("--output", required=True, help="Output PLY path")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument("--project-back", type=float, default=0.9)
    parser.add_argument("--scale-padding", type=float, default=1.05)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
        import torch
    except ImportError as e:
        print(f"ERROR: torch/numpy not importable: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        import cumesh
        import cumesh.remeshing  # noqa: F401
    except ImportError as e:
        print(f"ERROR: cumesh not importable in this Python interpreter: {e}", file=sys.stderr)
        sys.exit(2)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available in this torch build.", file=sys.stderr)
        sys.exit(3)

    if args.verbose:
        print(f"[worker] torch {torch.__version__}, CUDA ok, device={torch.cuda.get_device_name(0)}")
        print(f"[worker] loading {args.input}")

    try:
        np_vertices, np_faces = _read_ply(args.input)
    except Exception as e:
        print(f"ERROR: failed to read input PLY: {e}", file=sys.stderr)
        sys.exit(4)

    if np_vertices.shape[0] == 0 or np_faces.shape[0] == 0:
        print("ERROR: input PLY has no vertices or no faces.", file=sys.stderr)
        sys.exit(4)

    vertices = torch.from_numpy(np_vertices).float().cuda()
    faces = torch.from_numpy(np_faces.astype("int32")).int().cuda()

    if args.verbose:
        print(f"[worker] input: {vertices.shape[0]} verts, {faces.shape[0]} faces")

    aabb_max = vertices.max(dim=0)[0]
    aabb_min = vertices.min(dim=0)[0]
    center = (aabb_max + aabb_min) / 2
    scale = (aabb_max - aabb_min).max().item()

    if args.verbose:
        print(f"[worker] center={center.tolist()}, scale={scale}")
        print(f"[worker] resolution={args.resolution}, band={args.band}, "
              f"project_back={args.project_back}, scale_padding={args.scale_padding}")

    new_vertices, new_faces = cumesh.remeshing.remesh_narrow_band_dc(
        vertices, faces,
        center=center,
        scale=args.scale_padding * scale,
        resolution=args.resolution,
        band=args.band,
        project_back=args.project_back,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"[worker] output: {new_vertices.shape[0]} verts, {new_faces.shape[0]} faces")

    try:
        _write_ply(
            args.output,
            new_vertices.detach().cpu().numpy(),
            new_faces.detach().cpu().numpy(),
        )
    except Exception as e:
        print(f"ERROR: failed to write output PLY: {e}", file=sys.stderr)
        sys.exit(5)

    if args.verbose:
        print(f"[worker] wrote {args.output}")


if __name__ == "__main__":
    main()
