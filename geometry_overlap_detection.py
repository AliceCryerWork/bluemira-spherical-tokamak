# _summary_.
# Returns
# -------
#    _type_: _description_
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.reactor import Reactor
from bluemira.geometry.base import BoundingBox
from numpy.typing import NDArray
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from Part import Shape

PartShape: TypeAlias = "Shape"
IndexPairArray: TypeAlias = NDArray[np.int32]
FloatArray: TypeAlias = NDArray[np.float64]
NamedCollisionPair: TypeAlias = tuple[str, str]


class GeometryData(NamedTuple):
    """Names, boxes, shapes."""

    names: list[str]
    boxes: list[BoundingBox]
    shapes: list[PartShape]


class GeometryExtractor:
    """Flatten reactor geometry."""

    @staticmethod
    def extract(reactor: Reactor) -> GeometryData:
        def collect(root: Component) -> list[Component]:
            stack: list[Component] = [root]
            out: list[Component] = []
            while stack:
                node = stack.pop()
                out.append(node)
                stack.extend(node.children)
            return out

        # NOTE: Is there a public method for this?
        params = reactor._init_construction_param_values(None, {})  # noqa: SLF001
        tree = reactor._build_component_tree("xyz", params)  # noqa: SLF001

        physical_components: list[PhysicalComponent] = [
            c for c in collect(tree) if isinstance(c, PhysicalComponent)
        ]

        names: list[str] = []
        boxes: list[BoundingBox] = []
        shapes: list[PartShape] = []
        for i, c in enumerate(physical_components):
            names.append(f"{c.name}_{i}")
            boxes.append(c.shape.optimal_bounding_box)
            shapes.append(c.shape.shape)

        return GeometryData(names, boxes, shapes)


class BaseOverlapDetector(ABC):
    """Base Overlap Detector class."""

    @staticmethod
    def compute_bounding_box_centres(boxes: list[BoundingBox]) -> FloatArray:
        return np.array(
            [
                [
                    (b.x_min + b.x_max) * 0.5,
                    (b.y_min + b.y_max) * 0.5,
                    (b.z_min + b.z_max) * 0.5,
                ]
                for b in boxes
            ],
            dtype=float,
        )

    @staticmethod
    def collate_bounding_box_bounds(
        boxes: list[BoundingBox],
    ) -> tuple[FloatArray, FloatArray]:
        mins = np.array([[b.x_min, b.y_min, b.z_min] for b in boxes], dtype=float)
        maxs = np.array([[b.x_max, b.y_max, b.z_max] for b in boxes], dtype=float)
        return mins, maxs

    @staticmethod
    def remove_non_overlapping_pairs(
        pairs: IndexPairArray,
        mins: FloatArray,
        maxs: FloatArray,
        tol: float,
    ) -> IndexPairArray:
        if pairs.size == 0:
            return pairs

        i = pairs[:, 0]
        j = pairs[:, 1]

        ov = np.minimum(maxs[i], maxs[j]) - np.maximum(mins[i], mins[j])
        ov = np.maximum(ov, 0.0)

        vol = ov[:, 0] * ov[:, 1] * ov[:, 2]
        axis = np.min(ov, axis=1)

        return pairs[(vol > tol) & (axis > tol)]

    @staticmethod
    def determine_overlapping_shapes(
        pairs: IndexPairArray,
        shapes: list[PartShape],
        mins: FloatArray,
        maxs: FloatArray,
        tol: float,
    ) -> IndexPairArray:
        out: list[tuple[int, int]] = []

        for i, j in pairs:
            if (
                maxs[i, 0] < mins[j, 0]
                or mins[i, 0] > maxs[j, 0]
                or maxs[i, 1] < mins[j, 1]
                or mins[i, 1] > maxs[j, 1]
                or maxs[i, 2] < mins[j, 2]
                or mins[i, 2] > maxs[j, 2]
            ):
                continue

            common = shapes[i].common(shapes[j])
            if not common.isNull() and getattr(common, "Volume", 0.0) > tol:
                out.append((i, j))

        if not out:
            return np.empty((0, 2), dtype=np.int32)

        return np.array(out, dtype=np.int32)

    @classmethod
    @abstractmethod
    def detect(
        cls,
        geometry: GeometryData,
        tolerance: float = 1e-5,
    ) -> list[NamedCollisionPair]:
        """Abstract Method to be called when running the main detection routine."""
        return []


class KDTreeOverlapDetector(BaseOverlapDetector):
    """KDTree neighbour search."""

    @staticmethod
    def compute_bounding_sphere_radius(boxes: list[BoundingBox]) -> FloatArray:
        """Use to identifying neighbouring bounding boxes that could possibly collide."""
        ext = np.array(
            [
                [
                    b.x_max - b.x_min,
                    b.y_max - b.y_min,
                    b.z_max - b.z_min,
                ]
                for b in boxes
            ],
            dtype=float,
        )
        return 0.5 * np.linalg.norm(ext, axis=1)

    @classmethod
    def detect(
        cls,
        geometry: GeometryData,
        tolerance: float = 1e-5,
    ) -> list[NamedCollisionPair]:
        names, boxes, shapes = geometry

        centres = cls.compute_bounding_box_centres(boxes)
        mins, maxs = cls.collate_bounding_box_bounds(boxes)
        half = cls.compute_bounding_sphere_radius(boxes)

        if centres.shape[0] == 0:
            return []

        tree = KDTree(centres)
        gmax = float(np.max(half))

        cand: list[tuple[int, int]] = []

        for i, _ in enumerate(centres):
            nbrs = tree.query_ball_point(centres[i], half[i] + gmax)
            for j in nbrs:
                if i < j:
                    cand.append((i, j))  # noqa: PERF401

        if not cand:
            return []

        arr = np.unique(np.array(cand, dtype=np.int32), axis=0)
        arr = cls.remove_non_overlapping_pairs(arr, mins, maxs, tolerance)
        arr = cls.determine_overlapping_shapes(arr, shapes, mins, maxs, tolerance)

        return [(names[i], names[j]) for i, j in arr]


class SpatialGridOverlapDetector(BaseOverlapDetector):
    """Grid neighbour search."""

    @staticmethod
    def build_grid(
        centres: FloatArray,
        cell: float,
    ) -> dict[tuple[int, int, int], list[int]]:
        grid: dict[tuple[int, int, int], list[int]] = {}
        keys = np.floor(centres / cell).astype(int)

        for i, k in enumerate(keys):
            grid.setdefault((k[0], k[1], k[2]), []).append(i)

        return grid

    @staticmethod
    def grid_pairs(grid: dict[tuple[int, int, int], list[int]]) -> IndexPairArray:
        pairs: set[tuple[int, int]] = set()

        for (x, y, z), idx in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n = (x + dx, y + dy, z + dz)
                        if n not in grid:
                            continue

                        for i in idx:
                            for j in grid[n]:
                                if i < j:
                                    pairs.add((i, j))

        if not pairs:
            return np.empty((0, 2), dtype=np.int32)

        return np.array(list(pairs), dtype=np.int32)

    @classmethod
    def detect(
        cls,
        geometry: GeometryData,
        tolerance: float = 1e-5,
    ) -> list[NamedCollisionPair]:
        names, boxes, shapes = geometry

        centres = cls.compute_bounding_box_centres(boxes)
        mins, maxs = cls.collate_bounding_box_bounds(boxes)

        sizes = np.array([b.x_max - b.x_min for b in boxes], dtype=float)
        cell = float(np.mean(sizes) * 2.0)

        grid = cls.build_grid(centres, cell)
        arr = cls.grid_pairs(grid)
        arr = cls.remove_non_overlapping_pairs(arr, mins, maxs, tolerance)
        arr = cls.determine_overlapping_shapes(arr, shapes, mins, maxs, tolerance)

        return [(names[i], names[j]) for i, j in arr]


def fprint_overlaps(overlaps: Iterable[NamedCollisionPair]) -> None:
    """Use as helper function to print grouped overlaps in a human readable format."""
    counter: Counter[NamedCollisionPair] = Counter()
    pattern = re.compile(r"[\s_]*\d+(?:_\d+)*$")

    for first, second in overlaps:
        a = re.sub(pattern, "", first).strip()
        b = re.sub(pattern, "", second).strip()

        key: NamedCollisionPair = (a, b) if a <= b else (b, a)
        counter[key] += 1

    print(f"Total overlaps = {sum(counter.values())}")  # noqa: T201
    for (a, b), c in counter.items():
        print(f"{a}, {b}: {c}")  # noqa: T201


def bounding_box_from_shape(shape) -> BoundingBox:
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)

    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()

    return BoundingBox.from_xyz(
        np.array([x_min, x_max], dtype=float),
        np.array([y_min, y_max], dtype=float),
        np.array([z_min, z_max], dtype=float),
    )


def load_step(filepath: str) -> GeometryData:
    doc = TDocStd_Document("XmlXCAF")
    reader = STEPCAFControl_Reader()
    reader.ReadFile(filepath)
    reader.Transfer(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

    return extract_geometry_data(shape_tool)


def get_label_name(label):
    entry = TCollection_AsciiString()
    TDF_Tool.Entry(label, entry)
    return entry.ToCString()


def actual_label(label, shape_tool):
    if shape_tool.IsReference(label):
        ref = TDF_Label()
        shape_tool.GetReferredShape(label, ref)
        return ref
    return label


def traverse(
    label: TDF_Label, shape_tool: XCAFDoc_ShapeTool, names, boxes, shapes
) -> GeometryData:
    label = actual_label(label, shape_tool)

    occ_shape = shape_tool.GetShape(label)
    shape = PartShapeAdapter(occ_shape)
    shapes.append(shape)
    names.append(get_label_name(label))
    boxes.append(bounding_box_from_shape(shape.Shape))

    if shape_tool.IsAssembly(label):
        children = TDF_LabelSequence()
        shape_tool.GetComponents(label, children)

        for i in range(1, children.Length() + 1):
            traverse(
                children.Value(i),
                shape_tool,
                names,
                boxes,
                shapes,
            )


def extract_geometry_data(shape_tool: XCAFDoc_ShapeTool) -> GeometryData:
    names = []
    boxes = []
    shapes = []

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    for i in range(1, roots.Length() + 1):
        traverse(
            roots.Value(i),
            shape_tool,
            names,
            boxes,
            shapes,
        )

    return GeometryData(
        names=names,
        boxes=boxes,
        shapes=shapes,
    )


class PartShapeAdapter:
    """Class to adapt TopoDS_Compound to FreeCAD Part.Shape."""

    def __init__(self, shape: TopoDS_Shape):
        self.Shape = shape

    def common(self, other: "PartShapeAdapter") -> "PartShapeAdapter":
        op = BRepAlgoAPI_Common(self.Shape, other.Shape)
        op.Build()

        if not op.IsDone():
            return PartShapeAdapter(TopoDS_Shape())

        return PartShapeAdapter(op.Shape())

    def isNull(self) -> bool:  # noqa: N802
        return self.Shape.IsNull()

    @property
    def Volume(self) -> float:  # noqa: N802
        if self.Shape.IsNull():
            return 0.0

        props = GProp_GProps()

        try:
            brepgprop_VolumeProperties(self.Shape, props)
            return props.Mass()
        except RuntimeError:
            return 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a file with a specified tolerance."
    )

    parser.add_argument("filepath", type=str, help="Path to the input file")

    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-3,
        help="Tolerance value (default: %(default)s)",
    )

    args = parser.parse_args()

    filepath = args.filepath
    tolerance = args.tolerance

    geometry = load_step(filepath)
    kdtree = KDTreeOverlapDetector.detect(geometry, tolerance)
    spatial_grid = SpatialGridOverlapDetector.detect(geometry, tolerance)
    fprint_overlaps(kdtree)
    fprint_overlaps(spatial_grid)
