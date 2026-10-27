"""Generate the minimal 3D Tiles fixtures committed beside this script (not a test module).

Run it from the repo's dev environment to regenerate ``building.b3dm`` and ``tileset.json``:

    python tests/data/_make_3dtiles_fixture.py

The fixtures are hand-built minimal files (the repo convention — no large vendor samples in git):
a ``b3dm`` tile embedding an uncompressed glTF 2.0 mesh of two batched square prisms, each a
distinct ``_BATCHID`` with a batch-table row, plus a one-tile ``tileset.json`` that points at it.
Two features at known XY extents make the expected convex-hull footprints trivial to assert. The
binary layout follows the 3D Tiles 1.0 ``b3dm`` and glTF 2.0 ``glb`` specifications; the leading
underscore keeps pytest from collecting this generator as a test.
"""

import json
import struct
from pathlib import Path

HERE = Path(__file__).parent

# Two unit squares at known XY positions; z is the up axis (identity node transform), so each
# feature's convex-hull footprint is exactly its square. Feature 0 around (0..2, 0..2), feature 1
# around (5..7, 5..7).
_SQUARES = {
    0: [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    1: [(5.0, 5.0), (7.0, 5.0), (7.0, 7.0), (5.0, 7.0)],
}


def _pad(data: bytes, boundary: int, fill: bytes) -> bytes:
    """Pad ``data`` with ``fill`` up to the next multiple of ``boundary``."""
    remainder = len(data) % boundary
    return data if remainder == 0 else data + fill * (boundary - remainder)


def _build_glb() -> bytes:
    """A binary glTF holding the two batched squares as one POSITION + _BATCHID mesh."""
    positions: list[tuple[float, float, float]] = []
    batch_ids: list[int] = []
    for batch, corners in _SQUARES.items():
        for x, y in corners:
            positions.append((x, y, 0.0))
            batch_ids.append(batch)

    position_bytes = b"".join(struct.pack("<3f", *point) for point in positions)
    batch_bytes = b"".join(struct.pack("<H", value) for value in batch_ids)
    position_bytes = _pad(position_bytes, 4, b"\x00")
    binary = position_bytes + batch_bytes
    binary = _pad(binary, 4, b"\x00")

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "_BATCHID": 1}, "mode": 4}]}
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(position_bytes),
                "byteLength": len(batch_bytes),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(batch_ids),
                "type": "SCALAR",
            },
        ],
    }
    json_chunk = _pad(json.dumps(gltf).encode("utf-8"), 4, b" ")
    bin_chunk = _pad(binary, 4, b"\x00")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)
    json_part = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    bin_part = struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    return header + json_part + bin_part


def _build_b3dm(glb: bytes) -> bytes:
    """Wrap a glTF binary in a b3dm with a BATCH_LENGTH feature table and a JSON batch table."""
    feature_table = _pad(json.dumps({"BATCH_LENGTH": 2}).encode("utf-8"), 8, b" ")
    batch_table = _pad(
        json.dumps(
            {
                "id": ["b0", "b1"],
                "height": [10.0, 20.0],
                "type": ["Building", "Building"],
            }
        ).encode("utf-8"),
        8,
        b" ",
    )
    body = feature_table + batch_table + glb
    header = struct.pack(
        "<4sIIIIII",
        b"b3dm",
        1,
        28 + len(body),
        len(feature_table),
        0,
        len(batch_table),
        0,
    )
    return header + body


def main() -> None:
    """Write ``building.b3dm`` and ``tileset.json`` next to this script."""
    glb = _build_glb()
    b3dm = _build_b3dm(glb)
    (HERE / "building.b3dm").write_bytes(b3dm)
    tileset = {
        "asset": {"version": "1.0"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {"box": [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]},
            "geometricError": 0.0,
            "refine": "ADD",
            "content": {"uri": "building.b3dm"},
        },
    }
    (HERE / "tileset.json").write_text(json.dumps(tileset, indent=2), encoding="utf-8")
    print(f"wrote building.b3dm ({len(b3dm)} bytes) and tileset.json")


if __name__ == "__main__":
    main()
