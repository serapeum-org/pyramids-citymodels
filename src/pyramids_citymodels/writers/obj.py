"""Serialize the IR to Wavefront OBJ (feature #14 — export to meshes).

OBJ is the lowest-common-denominator mesh hand-off: a flat list of ``v`` vertices and ``f`` faces
that every 3-D tool reads. This writer walks every city object's geometries to surface level
(:func:`pyramids_citymodels.ir.iter_surfaces`, so it works for any geometry type — ``MultiSurface``,
``Solid``, …), de-duplicates the vertices into one shared pool, and emits each surface's **exterior
ring** as one face, grouped under an ``o <object-id>`` so a viewer can still tell the objects apart.

Deliberate, documented limits of this plain-geometry export:

- **Exterior rings only.** A surface's interior rings (holes) are dropped — OBJ faces have no native
  hole support; a hole would need the face triangulated around it, which is geometry processing this
  package leaves to a mesh tool.
- **Faces, not triangles.** Each polygon is emitted as a single n-gon ``f`` (OBJ allows it); a
  consumer that needs triangles triangulates on import. (The glTF writer, which *must* triangulate,
  does so.)
- **Appearance is opt-in** (``materials=True``) — off by default this is plain geometry (appearance
  round-trips through the CityJSON writer instead). With ``materials=True`` a companion ``.mtl`` sidecar
  carries each material's ``Kd`` diffuse colour (from the CityJSON ``diffuseColor``) and ``map_Kd``
  texture image, surfaces are grouped by material via ``usemtl`` (not by object), and a textured surface
  emits ``vt`` texture coordinates — but this needs a file path for the sidecar (it raises on the
  string-return form). The OBJ/MTL grammar is whitespace-delimited and has no escaping, so a ``map_Kd``
  image reference (or the derived ``.mtl`` name) containing a space is technically malformed — keep
  texture image paths and the output stem space-free.

Only the standard library is used, matching the other writers.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from pathlib import Path
from typing import Any

from pyramids_citymodels.ir import CityModel, CityObject, iter_surfaces
from pyramids_citymodels.writers import _appearance
from pyramids_citymodels.writers._geometry import open_ring as _open_ring
from pyramids_citymodels.writers._geometry import xyz as _xyz

__all__ = ["write_obj"]


class _VertexPool:
    """Intern ``(x, y, z)`` coordinates to 1-based OBJ vertex indices, de-duplicating repeats."""

    def __init__(self) -> None:
        self._index: dict[tuple[float, float, float], int] = {}
        self.vertices: list[tuple[float, float, float]] = []

    def intern(self, coordinate: Any) -> int:
        """Return the 1-based index of ``coordinate``, appending it on first sight."""
        key = _xyz(coordinate)
        existing = self._index.get(key)
        if existing is not None:
            return existing
        index = len(self.vertices) + 1  # OBJ indices are 1-based
        self._index[key] = index
        self.vertices.append(key)
        return index


def _object_faces(city_object: CityObject, pool: _VertexPool) -> list[list[int]]:
    """The OBJ faces of one city object — each surface's exterior ring interned to vertex indices.

    A surface whose exterior ring has fewer than three (opened) vertices is a degenerate non-face
    and is skipped.
    """
    faces: list[list[int]] = []
    for geometry in city_object.geometries:
        for surface, _value in iter_surfaces(
            geometry.boundaries, geometry.semantic_values
        ):
            ring = _open_ring(surface[0]) if surface else []
            if len(ring) < 3:
                continue
            faces.append([pool.intern(coordinate) for coordinate in ring])
    return faces


class _ObjPool:
    """Intern positions (1-based ``v``) and texture coordinates (1-based ``vt``) for a materialled OBJ."""

    def __init__(self) -> None:
        self.positions: list[tuple[float, float, float]] = []
        self.texcoords: list[tuple[float, float]] = []
        self._pos: dict[tuple[float, float, float], int] = {}
        self._uv: dict[tuple[float, float], int] = {}

    def position(self, coordinate: Any) -> int:
        key = _xyz(coordinate)
        index = self._pos.get(key)
        if index is None:
            index = len(self.positions) + 1
            self._pos[key] = index
            self.positions.append(key)
        return index

    def texcoord(self, uv: tuple[float, float]) -> int:
        index = self._uv.get(uv)
        if index is None:
            index = len(self.texcoords) + 1
            self._uv[uv] = index
            self.texcoords.append(uv)
        return index


def _mtl_entry(
    symbol: str, material_index: int | None, texture_index: int | None, model: CityModel
) -> tuple[list[str], bool]:
    """The ``.mtl`` lines for one material symbol, and whether it is textured (has a usable image).

    ``Kd`` is the material's diffuse colour (default grey when absent); ``map_Kd`` is the texture image
    reference when the texture resolves to one.
    """
    materials_pool = model.appearances.get("materials") or []
    textures_pool = model.appearances.get("textures") or []
    color = _appearance.base_color(
        _appearance.diffuse_of(material_index, materials_pool)
    )
    lines = [f"newmtl {symbol}", f"Kd {color[0]} {color[1]} {color[2]}"]
    image = _appearance.texture_image(texture_index, textures_pool)
    if image:
        lines.append(f"map_Kd {image}")
    return lines, image is not None


def _obj_documents(model: CityModel, mtl_name: str) -> tuple[str, str | None]:
    """Build the materialled ``(obj_text, mtl_text)`` — ``mtl_text`` is ``None`` when no material is used.

    Surfaces are grouped by their ``(material, texture)`` appearance; each appearance group gets a
    ``usemtl`` symbol (with ``vt`` texture coordinates when the texture resolves to an image). ``usemtl``
    is **stateful** in OBJ, so the plain ``(None, None)`` group is flushed **first** — under no material
    (the default), before any ``usemtl`` — otherwise its faces would inherit the preceding group's
    material. Faces are the surfaces' exterior rings (n-gons), positions and texcoords shared across the
    whole model.
    """
    uv_pool = model.appearances.get("vertices-texture") or []
    pool = _ObjPool()
    mtl_lines = ["# MTL generated by pyramids-citymodels"]
    material_count = (
        0  # each (material, texture) group is distinct, so a plain counter suffices
    )
    plain_body: list[
        str
    ] = []  # the (None, None) group — no usemtl, flushed first (default material)
    material_body: list[str] = []  # the materialled groups, each led by its own usemtl

    def face(ring: list[Any], uv_indices: list[int] | None, textured: bool) -> str:
        corners = []
        for vertex, coordinate in enumerate(ring):
            position = pool.position(coordinate)
            uv = _appearance.ring_uv(uv_pool, uv_indices, vertex) if textured else None
            corners.append(
                f"{position}/{pool.texcoord(uv)}" if uv is not None else str(position)
            )
        return "f " + " ".join(corners)

    for (material_index, texture_index), rings in _appearance.collect_groups(
        model
    ).items():
        if material_index is None and texture_index is None:
            plain_body.extend(
                face(ring, uv_indices, False) for ring, uv_indices in rings
            )
            continue
        symbol = f"mat_{material_count}"
        material_count += 1
        entry, textured = _mtl_entry(symbol, material_index, texture_index, model)
        mtl_lines.extend(entry)
        material_body.append(f"usemtl {symbol}")
        material_body.extend(
            face(ring, uv_indices, textured) for ring, uv_indices in rings
        )

    body = plain_body + material_body
    lines = ["# OBJ generated by pyramids-citymodels"]
    if material_count:
        lines.append(f"mtllib {mtl_name}")
    lines.extend(f"v {x} {y} {z}" for x, y, z in pool.positions)
    lines.extend(f"vt {u} {v}" for u, v in pool.texcoords)
    lines.extend(body)
    obj_text = "\n".join(lines) + "\n"
    return obj_text, ("\n".join(mtl_lines) + "\n" if material_count else None)


def write_obj(
    model: CityModel, path: Any = None, materials: bool = False
) -> str | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to Wavefront OBJ.

    With ``materials=False`` (default) each surface's exterior ring becomes one plain face; vertices are
    shared across the whole model and objects are written in document order, each preceded by an
    ``o <id>`` group marker.

    With ``materials=True`` the IR appearances ride along: surfaces are grouped by their
    ``(material, texture)`` into ``usemtl`` groups and a companion ``<stem>.mtl`` is written beside the
    ``.obj`` (referenced by ``mtllib``) carrying each material's ``Kd`` diffuse colour (from the CityJSON
    ``diffuseColor``) and ``map_Kd`` texture image; a textured surface also emits ``vt`` texture
    coordinates (from the ``vertices-texture`` pool) and ``f v/vt`` faces. Because the ``.mtl`` is a
    sidecar file, ``materials=True`` **requires a path** (it raises ``ValueError`` when ``path`` is
    ``None``) and groups by material rather than by object. A model with no appearances writes no
    material and no sidecar. See the module docstring for the geometry limits (exterior rings only,
    n-gon faces).

    Args:
        model: The city model to serialize.
        path: A filesystem path to write the ``.obj`` to, or ``None`` to return it as a string.
        materials: Carry IR appearances (a ``.mtl`` sidecar + UVs) instead of plain geometry; needs a path.

    Returns:
        The OBJ text as a ``str`` when ``path`` is ``None``; otherwise ``None`` (written to ``path``).

    Raises:
        ValueError: If ``materials=True`` and ``path`` is ``None`` (the ``.mtl`` sidecar needs a path).

    Examples:
        - A one-triangle building yields three ``v`` lines and one ``f`` under its ``o`` group:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> print(write_obj(model))
            # OBJ generated by pyramids-citymodels
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 1.0 1.0 0.0
            o b
            f 1 2 3
            <BLANKLINE>

            ```
        - A repeated vertex is pooled once, so two adjacent faces share indices:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> shared = (0.0, 0.0, 0.0)
            >>> a = [[shared, (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> b = [[shared, (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[a, b])])})
            >>> [line for line in write_obj(model).splitlines() if line.startswith("v ")]
            ['v 0.0 0.0 0.0', 'v 1.0 0.0 0.0', 'v 1.0 1.0 0.0', 'v 0.0 1.0 0.0']

            ```
    """
    if materials:
        if path is None:
            raise ValueError(
                "write_obj(materials=True) needs a path to write the .mtl sidecar beside the .obj"
            )
        target = Path(path)
        mtl_path = target.with_suffix(".mtl")
        obj_text, mtl_text = _obj_documents(model, mtl_path.name)
        target.write_text(obj_text, encoding="utf-8")
        if mtl_text is not None:
            mtl_path.write_text(mtl_text, encoding="utf-8")
        return None

    pool = _VertexPool()
    faces_by_object = [
        (object_id, _object_faces(city_object, pool))
        for object_id, city_object in model.objects.items()
    ]

    lines = ["# OBJ generated by pyramids-citymodels"]
    lines.extend(f"v {x} {y} {z}" for x, y, z in pool.vertices)
    for object_id, faces in faces_by_object:
        if not faces:
            continue
        lines.append(f"o {object_id}")
        lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    text = "\n".join(lines) + "\n"

    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return None
