import numpy as np
import pandas as pd
import heapq
from typing import Callable

def euclidean_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    return np.sqrt(((dot_x - dot_y) ** 2).sum())

def manhattan_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    return np.abs((dot_x - dot_y)).sum()

def cos_dist(dot_x : np.ndarray, dot_y : np.ndarray):
    return 1 - dot_x.dot(dot_y) / (np.linalg.norm(dot_x) * np.linalg.norm(dot_y))

k_near_dots = 11
obj_param = pd.read_csv("4-classes_1000x2/original.csv", sep = "," ,encoding = "utf-8")
obj_classes = pd.read_csv("4-classes_1000x2/class.csv", sep = "," ,encoding = "utf-8")

test_dot = np.array([1, 2], dtype = "float64")
# test_dot = np.random.uniform(-6, 6, size = obj_param.shape[1]) # Способ задания тестовой точки случайным образом

"""
K-D Tree, Data Structure Realization
"""

class KDNode:
    def __init__(self, point : np.ndarray | None = None, axis : int = 0,
                 point_class : int | None = None):
        self.point : np.ndarray | None = point
        self.axis : int = axis
        self.left : KDNode | None = None
        self.right : KDNode | None = None
        self.point_class : int | None = point_class

class KDTree:
    def __init__(
        self,
        func_dist: Callable,
        points: pd.DataFrame | None = None,
        dim: int | None = None,
        point_classes: list | None = None
    ) -> None:
        if dim is None:
            dim = len(points.iloc[0]) if not(points is None) else 2
        self.func_dist: Callable = func_dist
        self.dim : int = dim
        self.root: KDNode | None = None
        self.size = 0
        if not(points is None):
            point_classes = point_classes if not(point_classes is None) else [None for _ in range(len(points))]
            items = [(np.array(points.iloc[i]), point_classes[i]) for i in range(len(points))]
            self.root = self._build(items, depth = 0)
            self.size = len(points)

    def _build(
        self, items: list, depth: int
    ) -> KDNode | None:
        if len(items) == 0:
            return None

        axis = depth % self.dim
        """
        axis - по какому измерению мы сейчас делим 'плоскость'
        """
        items.sort(key = lambda p : p[0][axis])
        mid = len(items) // 2

        node = KDNode(point = items[mid][0], axis = axis, point_class = items[mid][1])
        node.left = self._build(items[ : mid], depth + 1)
        node.right = self._build(items[mid :], depth + 1)
        return node

    def rebuild(self) -> None:
        """Flatten the tree and rebuild it balanced. Useful after many
        insertions/deletions have skewed the tree."""
        items = [(n.point, n.data) for n in self._inorder_nodes()]
        self.root = self._build(items, depth=0)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def insert(self, point: np.ndarray, point_class: int | None = None) -> None:
        self.size += 1
        if self.root is None:
            self.root = KDNode(point = point, axis = 0, point_class = point_class)
            return
        self._insert(self.root, point, depth = 0, point_class = point_class)

    def _insert(self, node: KDNode, point: np.ndarray, depth: int, point_class: int | None = None) -> None:
        axis = node.axis
        if point[axis] < node.point[axis]:
            if node.left is None:
                node.left = KDNode(point = point, axis = (depth + 1) % self.dim, point_class = point_class)
            else:
                self._insert(node.left, point, depth + 1, point_class)
        else:
            if node.right is None:
                node.right = KDNode(point = point, axis = (depth + 1) % self.dim, point_class = point_class)
            else:
                self._insert(node.right, point, depth + 1, point_class)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete(self, point: np.ndarray) -> bool:
        """Delete a point from the tree. Returns True if a point was
        removed, False if the point was not found."""
        self.root, removed = self._delete(self.root, point, depth=0)
        if removed:
            self.size -= 1
        return removed

    def _delete(
        self, node: KDNode, point: np.ndarray, depth: int
    ) -> tuple:
        if node is None:
            return None, False

        if node.point == point:
            axis = node.axis
            if node.right is not None:
                successor = self._find_min(node.right, axis, depth + 1)
                node.point, node.data = successor.point, successor.data
                node.right, _ = self._delete(node.right, successor.point, depth + 1)
            elif node.left is not None:
                successor = self._find_min(node.left, axis, depth + 1)
                node.point, node.data = successor.point, successor.data
                # Per Bentley's algorithm, move the left subtree to the right
                node.right, _ = self._delete(node.left, successor.point, depth + 1)
                node.left = None
            else:
                return None, True
            return node, True

        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left, removed = self._delete(node.left, point, depth + 1)
        else:
            node.right, removed = self._delete(node.right, point, depth + 1)
        return node, removed

    def _find_min(self, node: KDNode, target_axis: int, depth: int) -> KDNode:
        """Find the node with the minimum coordinate along `target_axis`
        in the subtree rooted at `node` (used by delete)."""
        if node is None:
            return None
        axis = node.axis
        if axis == target_axis:
            if node.left is None:
                return node
            return self._find_min(node.left, target_axis, depth + 1)
        left_min = self._find_min(node.left, target_axis, depth + 1)
        right_min = self._find_min(node.right, target_axis, depth + 1)
        candidates = [n for n in (node, left_min, right_min) if n is not None]
        return min(candidates, key=lambda n: n.point[target_axis])

    # ------------------------------------------------------------------
    # Nearest neighbour (1-NN)
    # ------------------------------------------------------------------
    def nearest_neighbor(self, target: Point) -> Optional[Tuple[Tuple[float, ...], Any, float]]:
        """Return (point, data, distance) of the closest point to target,
        or None if the tree is empty."""
        if self.root is None:
            return None
        best = {"node": None, "dist": math.inf}
        self._nn_search(self.root, tuple(map(float, target)), best)
        node = best["node"]
        return (node.point, node.data, math.sqrt(best["dist"])) if node else None

    def _nn_search(self, node: Optional[KDNode], target: Tuple[float, ...], best: dict) -> None:
        if node is None:
            return
        d = self.func_dist(node.point, target)
        if d < best["dist"]:
            best["dist"] = d
            best["node"] = node

        axis = node.axis
        diff = target[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._nn_search(near, target, best)
        # Only descend into the far subtree if the splitting hyperplane
        # is closer than the current best distance (branch-and-bound pruning)
        if diff * diff < best["dist"]:
            self._nn_search(far, target, best)

    # ------------------------------------------------------------------
    # k-nearest neighbours
    # ------------------------------------------------------------------
    def k_nearest_neighbors(
        self, target: Point, k: int
    ) -> List[Tuple[Tuple[float, ...], Any, float]]:
        """Return the k closest points to target as a list of
        (point, data, distance), sorted from nearest to farthest."""
        target = tuple(map(float, target))
        heap: List[Tuple[float, int, KDNode]] = []  # max-heap via negated distance
        counter = 0  # tie-breaker so heapq never compares KDNode objects

        def visit(node: Optional[KDNode]) -> None:
            nonlocal counter
            if node is None:
                return
            d = _sq_distance(node.point, target)
            if len(heap) < k:
                heapq.heappush(heap, (-d, counter, node))
                counter += 1
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, counter, node))
                counter += 1

            axis = node.axis
            diff = target[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            if len(heap) < k or diff * diff < -heap[0][0]:
                visit(far)

        visit(self.root)
        result = [(node.point, node.data, math.sqrt(-neg_d)) for neg_d, _, node in heap]
        result.sort(key=lambda r: r[2])
        return result

    # ------------------------------------------------------------------
    # Orthogonal range search
    # ------------------------------------------------------------------
    def range_search(
        self, lower: Point, upper: Point
    ) -> List[Tuple[Tuple[float, ...], Any]]:
        """Return every point p such that lower[i] <= p[i] <= upper[i]
        for every dimension i (an axis-aligned hyper-rectangle query)."""
        lower = tuple(map(float, lower))
        upper = tuple(map(float, upper))
        results: List[Tuple[Tuple[float, ...], Any]] = []
        self._range_search(self.root, lower, upper, results)
        return results

    def _range_search(
        self,
        node: Optional[KDNode],
        lower: Tuple[float, ...],
        upper: Tuple[float, ...],
        results: List[Tuple[Tuple[float, ...], Any]],
    ) -> None:
        if node is None:
            return
        if all(lower[i] <= node.point[i] <= upper[i] for i in range(self.k)):
            results.append((node.point, node.data))

        axis = node.axis
        if lower[axis] <= node.point[axis]:
            self._range_search(node.left, lower, upper, results)
        if upper[axis] >= node.point[axis]:
            self._range_search(node.right, lower, upper, results)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def height(self) -> int:
        def _h(node: Optional[KDNode]) -> int:
            if node is None:
                return 0
            return 1 + max(_h(node.left), _h(node.right))

        return _h(self.root)

    def _inorder_nodes(self) -> List[KDNode]:
        out: List[KDNode] = []

        def _walk(node: Optional[KDNode]) -> None:
            if node is None:
                return
            _walk(node.left)
            out.append(node)
            _walk(node.right)

        _walk(self.root)
        return out

    def all_points(self) -> List[Tuple[Tuple[float, ...], Any]]:
        return [(n.point, n.data) for n in self._inorder_nodes()]