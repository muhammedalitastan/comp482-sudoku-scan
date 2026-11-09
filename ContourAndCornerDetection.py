"""Stage 2 - Harris corner detection and outer-corner selection.

cornerHarris -> local maxima -> greedy NMS -> optional contour region
proposal -> match each vertex to the nearest Harris maximum -> cornerSubPix.
Final corner coordinates always come from the Harris response map; contours
only tell us where to look.
"""

import cv2
import numpy as np
from dataclasses import dataclass


class CornerNotFoundError(Exception):
    """Raised when no valid quadrilateral can be selected from Harris corners."""


@dataclass
class HarrisParams:
    block_size: int = 2
    ksize: int = 3
    k: float = 0.04
    threshold_ratio: float = 0.01  # fraction of the maximum Harris response
    nms_distance: int | None = None  # None -> 1% of the smallest image side
    max_candidates: int = 2000
    vertex_radius_ratio: float = 0.015  # search radius around a region vertex
    vertex_floor_ratio: float = 0.002  # min Harris response for vertex matching


@dataclass
class HarrisDetection:
    corners: np.ndarray | None  # [TL, TR, BR, BL] as float32, or None
    harris_vis: np.ndarray
    candidates: np.ndarray
    message: str = ""
    method: str = ""  # "contour+harris" or "harris-rule"


def _local_maxima(response: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Pixel coords of local maxima above threshold, strongest first."""
    size = 2 * min_distance + 1
    kernel = np.ones((size, size), np.uint8)
    dilated = cv2.dilate(response, kernel)
    mask = (response >= dilated) & (response >= threshold)
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    order = np.argsort(response[ys, xs])[::-1]
    return points[order]


def _non_maximum_suppression(corners: np.ndarray, min_distance: int) -> np.ndarray:
    """Greedy NMS: drop corners within min_distance of a stronger kept one."""
    if len(corners) == 0:
        return corners
    keep = []
    suppressed = np.zeros(len(corners), dtype=bool)
    for i, corner in enumerate(corners):
        if suppressed[i]:
            continue
        keep.append(corner)
        if i + 1 < len(corners):
            dist = np.linalg.norm(corners[i + 1:] - corner, axis=1)
            suppressed[i + 1:] |= dist < min_distance
    return np.array(keep, dtype=np.float32)


def _find_region_vertices(region_map: np.ndarray):
    """Region only: largest contour approximated to 4 vertices. Not a corner source."""
    contours, _ = cv2.findContours(region_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    image_area = region_map.shape[0] * region_map.shape[1]
    if cv2.contourArea(largest) < 0.02 * image_area:
        return None

    perimeter = cv2.arcLength(largest, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(largest, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _extreme_index(values: np.ndarray, mode: str, exclude: set) -> int:
    """Index of the min/max value, skipping indices in exclude."""
    order = np.argsort(values) if mode == "min" else np.argsort(-values)
    for idx in order:
        if int(idx) not in exclude:
            return int(idx)
    raise CornerNotFoundError("not enough distinct points")


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL] using the x+y / x-y rule."""
    sum_scores = points[:, 0] + points[:, 1]
    diff_scores = points[:, 0] - points[:, 1]
    tl = _extreme_index(sum_scores, "min", set())
    br = _extreme_index(sum_scores, "max", {tl})
    tr = _extreme_index(diff_scores, "max", {tl, br})
    bl = _extreme_index(diff_scores, "min", {tl, br, tr})
    return points[[tl, tr, br, bl]]


def _match_vertices(vertices: np.ndarray, response: np.ndarray, radius: float, floor: float):
    """Nearest Harris maximum to each ordered vertex, inside a (2*radius) window."""
    h, w = response.shape
    kernel = np.ones((5, 5), np.uint8)
    matched = []

    for vertex in vertices:
        x0, x1 = max(0, int(vertex[0] - radius)), min(w, int(vertex[0] + radius))
        y0, y1 = max(0, int(vertex[1] - radius)), min(h, int(vertex[1] + radius))
        window = response[y0:y1, x0:x1]

        dilated = cv2.dilate(window, kernel)
        mask = (window >= dilated) & (window >= floor)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        points = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
        dists = np.linalg.norm(points - vertex, axis=1)
        matched.append(points[int(np.argmin(dists))])

    return np.array(matched, dtype=np.float32)


def _quad_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_convex(points: np.ndarray) -> bool:
    signs = []
    n = len(points)
    for i in range(n):
        a, b, c = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            signs.append(np.sign(cross))
    return len(signs) <= 1 or all(s == signs[0] for s in signs)


def _validate_quadrilateral(points: np.ndarray, image_shape) -> None:
    h, w = image_shape
    area = _quad_area(points)
    if area < 0.02 * h * w:
        raise CornerNotFoundError(
            f"selected corners form a quadrilateral that is too small ({area:.0f} px^2)"
        )
    if not _is_convex(points):
        raise CornerNotFoundError("selected corners do not form a convex quadrilateral")


def select_outer_corners(corners: np.ndarray, image_shape) -> np.ndarray:
    """
    Fallback: pick the 4 outer corners by the x+y / x-y rule and validate
    the quadrilateral (convex, >= 2% of image area). Returns [TL,TR,BR,BL].
    """
    if len(corners) < 4:
        raise CornerNotFoundError(
            f"only {len(corners)} Harris corners found, need at least 4"
        )

    ordered = _order_quad_points(corners)
    _validate_quadrilateral(ordered, image_shape)
    return ordered.astype(np.float32)


def _refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel refinement of the 4 selected corners."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    points = corners.reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def find_sudoku_grid(
    gray: np.ndarray,
    region_map: np.ndarray | None = None,
    params: HarrisParams | None = None,
) -> HarrisDetection:
    """Detect the 4 Sudoku corners. corners is None with a message on failure."""
    params = params or HarrisParams()
    h, w = gray.shape

    response = cv2.cornerHarris(gray, params.block_size, params.ksize, params.k)
    threshold = params.threshold_ratio * float(response.max())

    min_distance = params.nms_distance or max(5, int(min(h, w) * 0.01))
    candidates = _local_maxima(response, threshold, min_distance)
    candidates = candidates[: params.max_candidates]
    candidates = _non_maximum_suppression(candidates, min_distance)

    harris_vis = cv2.cvtColor(
        cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    for corner in candidates:
        cv2.circle(harris_vis, (int(corner[0]), int(corner[1])), 2, (0, 255, 0), -1)

    corners = None
    message = ""
    method = ""

    # Region finding (contour) -> match Harris candidates to its vertices
    region_vertices = None
    if region_map is not None:
        region_vertices = _find_region_vertices(region_map)
        if region_vertices is not None and not _is_convex(region_vertices):
            region_vertices = None

    if region_vertices is not None:
        cv2.polylines(
            harris_vis,
            [region_vertices.astype(np.int32).reshape(-1, 1, 2)],
            True, (255, 0, 255), 1,
        )
        ordered_vertices = _order_quad_points(region_vertices)
        radius = max(8, int(min(h, w) * params.vertex_radius_ratio))
        floor = params.vertex_floor_ratio * float(response.max())
        matched = _match_vertices(ordered_vertices, response, radius, floor)
        if matched is not None:
            try:
                _validate_quadrilateral(matched, gray.shape)
                corners = _refine_corners(gray, matched)
                method = "contour+harris"
            except CornerNotFoundError as exc:
                message = str(exc)
        else:
            message = "no Harris corner found near a grid vertex"
    else:
        # Fallback: pure Harris rule over all candidates
        try:
            corners = select_outer_corners(candidates, gray.shape)
            corners = _refine_corners(gray, corners)
            method = "harris-rule"
        except CornerNotFoundError as exc:
            message = str(exc)

    if corners is None:
        return HarrisDetection(None, harris_vis, candidates, message)

    colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0)]
    labels = ["TL", "TR", "BR", "BL"]
    for i, corner in enumerate(corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(harris_vis, (x, y), 8, colors[i], 2)
        cv2.putText(
            harris_vis, labels[i], (x - 14, y - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2,
        )

    return HarrisDetection(corners, harris_vis, candidates, method=method)
