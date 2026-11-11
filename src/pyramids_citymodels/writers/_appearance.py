"""Shared appearance walking for the mesh writers (OBJ / COLLADA / glTF) — feature #14 / #16.

The mesh writers export plain geometry by default and, opt-in (``materials=True``), carry the IR
appearances along. This module holds the format-neutral half of that: it groups every exterior ring by
its CityJSON ``(material, texture)`` so each writer emits one material group per distinct appearance,
and it turns a CityJSON material/texture into renderer-ready values — an RGBA base colour from
``diffuseColor`` and a texture's per-vertex UV indices into the ``vertices-texture`` pool. Each writer
then renders a group in its own syntax (glTF primitives, OBJ ``usemtl`` groups + ``.mtl``, COLLADA
``<triangles material>`` + ``<library_materials>``).

Everything here is defensive: a malformed appearance (wrong-length ``diffuseColor``, a non-2D /
out-of-range UV, a non-dict pool entry, a UV/ring count mismatch) degrades to the default colour or to
untextured rather than raising, so a bad source file never aborts a mesh export. No third-party
dependencies (only the standard library and this package).
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import base64
import math
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from pyramids_citymodels.ir import CityModel, Geometry, iter_surfaces
from pyramids_citymodels.writers._geometry import open_ring, xyz

#: A fully opaque default base colour for a material with no usable ``diffuseColor``.
DEFAULT_BASE_COLOR = [0.8, 0.8, 0.8, 1.0]

#: One exterior ring ready to mesh: its xyz vertices and (when textured) a UV index per vertex.
Ring = tuple[list[tuple[float, float, float]], "list[int] | None"]
#: The appearance key grouping surfaces into one material group: ``(material index, texture index)``.
GroupKey = tuple["int | None", "int | None"]


def appearance_values(geometry: Geometry, kind: str, default_theme: Any) -> Any:
    """The ``material`` / ``texture`` ``values`` for a geometry's chosen theme (``None`` if it has none).

    ``kind`` is ``"material"`` or ``"texture"``. The document's ``default-theme-*`` is used when the
    geometry carries it; otherwise the geometry's first theme is taken, so a single-theme model needs no
    configuration. The returned structure mirrors the geometry's boundaries (one leaf per surface), so
    :func:`~pyramids_citymodels.ir.iter_surfaces` pairs it with each surface in lock-step.
    """
    block = (geometry.appearance or {}).get(kind)
    if not block:
        return None
    theme = default_theme if default_theme in block else next(iter(block))
    return (block.get(theme) or {}).get("values")


def is_uv(point: Any) -> bool:
    """True when ``point`` is a 2-component finite real UV entry (packable, no ``bool``).

    ``bool`` is excluded though it is an ``int`` subclass (a boolean is not a texture coordinate), and a
    non-finite component (``NaN`` / ``inf``) is rejected — those render as malformed ``nan`` / ``inf``
    tokens in the text mesh writers, so a surface with such a UV degrades to untextured instead.
    """
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(
            isinstance(component, (int, float))
            and not isinstance(component, bool)
            and math.isfinite(component)
            for component in point
        )
    )


def texture_for_ring(
    texture_leaf: Any, ring_length: int, uv_pool: list[Any]
) -> tuple[int | None, list[int] | None]:
    """A surface's ``(texture index, UV indices)`` from its CityJSON texture leaf, or ``(None, None)``.

    A CityJSON texture value for a surface is ``[[texture_index, uv0, uv1, ...], ...]`` — one list per
    ring, whose head is the texture index and whose tail is a UV index per ring vertex. Only the
    exterior ring is meshed, the UV count must match the ring, and every UV index must point at a
    2-component entry within the ``vertices-texture`` pool; otherwise the surface is treated as
    untextured, so a malformed, partial, out-of-range or non-2D texture never corrupts the mesh.
    """
    if not isinstance(texture_leaf, list) or not texture_leaf:
        return None, None
    entry = texture_leaf[0]
    if not isinstance(entry, list) or not entry or not isinstance(entry[0], int):
        return None, None
    uv_indices = entry[1:]
    if len(uv_indices) != ring_length or not all(
        isinstance(i, int) for i in uv_indices
    ):
        return None, None
    if not all(
        0 <= index < len(uv_pool) and is_uv(uv_pool[index]) for index in uv_indices
    ):
        return None, None
    return entry[0], uv_indices


def ring_group(
    surface: Any, material_leaf: Any, texture_leaf: Any, uv_pool: list[Any]
) -> tuple[GroupKey, Ring] | None:
    """The ``(group key, ring)`` for one surface, or ``None`` when it has no meshable exterior ring.

    The group key is ``(material index, texture index)``; ``None`` for either when the surface carries
    no (valid) material/texture.
    """
    if not surface:
        return None
    ring = [xyz(coordinate) for coordinate in open_ring(surface[0])]
    if len(ring) < 3:
        return None
    material_index = material_leaf if isinstance(material_leaf, int) else None
    texture_index, uv_indices = texture_for_ring(texture_leaf, len(ring), uv_pool)
    return (material_index, texture_index), (ring, uv_indices)


def collect_groups(model: CityModel) -> dict[GroupKey, list[Ring]]:
    """Group every exterior ring in the model by its ``(material index, texture index)`` appearance key.

    Each group becomes one material group in the output, so a textured model emits as few groups as it
    has distinct material/texture combinations. A surface with no material/texture lands under
    ``(None, None)`` — the plain-geometry group.
    """
    groups: dict[GroupKey, list[Ring]] = {}
    default_mat = model.appearances.get("default-theme-material")
    default_tex = model.appearances.get("default-theme-texture")
    uv_pool = model.appearances.get("vertices-texture") or []
    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            material_values = appearance_values(geometry, "material", default_mat)
            texture_values = appearance_values(geometry, "texture", default_tex)
            paired = zip(
                iter_surfaces(geometry.boundaries, material_values),
                iter_surfaces(geometry.boundaries, texture_values),
            )
            for (surface, material_leaf), (_surface, texture_leaf) in paired:
                grouped = ring_group(surface, material_leaf, texture_leaf, uv_pool)
                if grouped is not None:
                    key, ring = grouped
                    groups.setdefault(key, []).append(ring)
    return groups


def ring_uv(uv_pool: list[Any], uv_indices: list[int] | None, vertex: int) -> Any:
    """The ``(u, v)`` for a ring vertex from the pool, or ``None`` when the ring is untextured."""
    return tuple(uv_pool[uv_indices[vertex]]) if uv_indices is not None else None


def diffuse_of(material_index: int | None, materials_pool: list[Any]) -> Any:
    """The CityJSON ``diffuseColor`` of a material pool entry, or ``None`` (missing / out-of-range / non-dict)."""
    if material_index is None or not (0 <= material_index < len(materials_pool)):
        return None
    entry = materials_pool[material_index]
    return entry.get("diffuseColor") if isinstance(entry, dict) else None


def base_color(diffuse: Any) -> list[float]:
    """A valid 4-component RGBA base colour (each in ``[0, 1]``) from a CityJSON ``diffuseColor``.

    A ``diffuseColor`` that is not a non-empty numeric sequence falls back to the default opaque grey;
    otherwise its first three components (padded with 0.0 if short) are clamped to ``[0, 1]`` and an
    opaque alpha appended. CityJSON ``diffuseColor`` is RGB, so alpha is always ``1.0`` and a 4th-or-later
    component of a non-spec value is ignored (a NaN clamps to 0.0 — no non-finite ever leaks).
    """
    if (
        not isinstance(diffuse, (list, tuple))
        or not diffuse
        or not all(isinstance(component, (int, float)) for component in diffuse)
    ):
        return list(DEFAULT_BASE_COLOR)
    rgb = ([float(component) for component in diffuse] + [0.0, 0.0, 0.0])[:3]
    return [min(1.0, max(0.0, component)) for component in rgb] + [1.0]


def texture_image(texture_index: int | None, textures_pool: list[Any]) -> str | None:
    """The CityJSON ``image`` reference for a texture pool entry, or ``None`` (missing / non-dict)."""
    if texture_index is None or not (0 <= texture_index < len(textures_pool)):
        return None
    entry = textures_pool[texture_index]
    image = entry.get("image") if isinstance(entry, dict) else None
    return image or None


def resolve_image(image: str, base: Any) -> Path | None:
    """A readable local image :class:`~pathlib.Path` for a CityJSON ``image``, or ``None``.

    Used by the ``embed_textures`` option: an absolute URL (``scheme://``) is not read (no network I/O),
    and a path that does not resolve to a readable file returns ``None`` so the caller falls back to a
    plain URI reference. Relative paths resolve against ``base`` (the texture base dir, or the cwd).

    Note: ``embed_textures`` reads whatever local file the CityJSON ``image`` field points at, including
    an **absolute** path (which ignores ``base``) or one that escapes ``base`` with ``..`` — the document's
    own texture references are trusted. It is opt-in, so embed only models whose ``image`` paths you
    trust; leave it off (the default, reference-only) to avoid reading any image file.
    """
    if not image or "://" in str(image):
        return None
    try:
        path = Path(base) / image if base is not None else Path(image)
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def data_uri(path: Path) -> str:
    """A base64 ``data:`` URI for an image file (its media type guessed from the extension)."""
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class ImageCopier:
    """Copies texture images beside an output file, disambiguating basename collisions.

    Dedupes by resolved source path (one source is copied once), and never overwrites a file it did not
    write (a pre-existing file, or a different source that shares a basename, is given a disambiguated
    ``<stem>_<n><suffix>`` name instead). So two textures named ``wall.png`` from different directories —
    or an output directory that already holds an unrelated ``wall.png`` — don't clobber each other.
    """

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._by_source: dict[Path, str] = {}

    def copy(self, source: Path) -> str:
        """Copy ``source`` beside the output under a collision-free basename and return that basename."""
        resolved = source.resolve()
        if resolved in self._by_source:
            return self._by_source[resolved]
        # The source already sits beside the output under its own name — reference it in place (no copy).
        if resolved == (self._out_dir / source.name).resolve():
            self._by_source[resolved] = source.name
            return source.name
        name = self._unique_name(source)
        shutil.copyfile(source, self._out_dir / name)
        self._by_source[resolved] = name
        return name

    def _unique_name(self, source: Path) -> str:
        assigned = set(self._by_source.values())
        candidate = source.name
        counter = 1
        # Avoid a name already assigned this run, and any file already in the output dir we didn't write.
        while candidate in assigned or (self._out_dir / candidate).exists():
            candidate = f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        return candidate
