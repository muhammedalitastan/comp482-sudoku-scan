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


# A puzzle quad must be large enough to be the grid and small enough not to be
# the whole frame. The side-ratio gate rejects slivers that still pass the area test.
_MIN_AREA_RATIO = 0.02
_MAX_AREA_RATIO = 0.98
_MAX_SIDE_RATIO = 5.0


@dataclass
class HarrisParams:
    block_size: int = 2
    ksize: int = 3
    k: float = 0.04
    threshold_ratio: float = 0.005  # fraction of the maximum Harris response
    nms_distance: int | None = None  # None -> 1% of the smallest image side
    max_candidates: int = 2000
    vertex_radius_ratio: float = 0.025  # search radius around a region vertex
    vertex_floor_ratio: float = 0.0002  # min Harris response for vertex matching


@dataclass
class HarrisDetection:
    corners: np.ndarray | None  # [TL, TR, BR, BL] as float32, or None
    harris_vis: np.ndarray
    candidates: np.ndarray
    message: str = ""
    method: str = ""  # contour+harris, region-bbox+harris, or harris-rule


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
    """Region only: a contour approximated to 4 vertices. Not a corner source.

    Contours are tried from largest to smallest. A contour that fills the frame
    is skipped, and contours below the minimum puzzle area stop the search.
    """
    contours, _ = cv2.findContours(region_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(region_map.shape[0] * region_map.shape[1])
    if image_area <= 0:
        return None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area > _MAX_AREA_RATIO * image_area:
            continue
        if area < _MIN_AREA_RATIO * image_area:
            break
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) != 4:
                continue
            points = approx.reshape(4, 2).astype(np.float32)
            if _is_convex(points):
                return points
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
    """True when consecutive edges turn the same way and the quad is not flat.

    A zero cross product (three collinear points) is ignored once. Fewer than
    three nonzero turns means the quad has collapsed, so it is rejected.
    """
    signs = []
    n = len(points)
    for i in range(n):
        a, b, c = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            signs.append(np.sign(cross))
    return len(signs) >= 3 and all(s == signs[0] for s in signs)


def _side_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)


def _validate_quadrilateral(points: np.ndarray, image_shape) -> None:
    h, w = image_shape
    area = _quad_area(points)
    image_area = float(h * w)
    if area < _MIN_AREA_RATIO * image_area:
        raise CornerNotFoundError(
            f"selected corners form a quadrilateral that is too small ({area:.0f} px^2)"
        )
    if area > _MAX_AREA_RATIO * image_area:
        raise CornerNotFoundError(
            f"selected corners cover too much of the image ({area:.0f} px^2)"
        )
    if not _is_convex(points):
        raise CornerNotFoundError("selected corners do not form a convex quadrilateral")
    sides = _side_lengths(points)
    shortest = float(np.min(sides))
    if shortest <= 1.0:
        raise CornerNotFoundError("selected corners are not separated")
    ratio = float(np.max(sides) / shortest)
    if ratio > _MAX_SIDE_RATIO:
        raise CornerNotFoundError(
            f"selected quadrilateral is too elongated (side ratio {ratio:.2f})"
        )


def select_outer_corners(corners: np.ndarray, image_shape) -> np.ndarray:
    """
    Fallback: pick the 4 outer corners by the x+y / x-y rule and validate
    the quadrilateral (strictly convex, area within 2%..98% of the image,
    longest side at most 5x the shortest). Returns [TL,TR,BR,BL].
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


def _candidates_in_bbox(candidates: np.ndarray, vertices: np.ndarray, image_shape, pad_ratio: float = 0.03):
    """Harris candidates inside the region box, expanded by a fraction of the short side."""
    if len(candidates) == 0:
        return candidates
    h, w = image_shape
    pad = pad_ratio * float(min(h, w))
    x0, y0 = vertices.min(axis=0) - pad
    x1, y1 = vertices.max(axis=0) + pad
    mask = (
        (candidates[:, 0] >= x0) & (candidates[:, 0] <= x1)
        & (candidates[:, 1] >= y0) & (candidates[:, 1] <= y1)
    )
    return candidates[mask]


def _corners_from_candidates(candidates, image_shape, refine_image):
    """x+y / x-y selection plus sub-pixel refinement, or (None, message)."""
    try:
        ordered = select_outer_corners(candidates, image_shape)
    except CornerNotFoundError as exc:
        return None, str(exc)
    return _refine_corners(refine_image, ordered), ""


def find_sudoku_grid(
    gray: np.ndarray,
    region_map: np.ndarray | None = None,
    params: HarrisParams | None = None,
    refine_image: np.ndarray | None = None,
) -> HarrisDetection:
    """Detect the 4 Sudoku corners. corners is None with a message on failure.

    `gray` is the image Harris runs on (the denoised frame). `refine_image`,
    when given, is the sharp grayscale used only by cornerSubPix.
    """
    params = params or HarrisParams()
    if refine_image is None or refine_image.shape[:2] != gray.shape[:2]:
        refine_image = gray
    h, w = gray.shape

    if params.block_size < 2:
        raise ValueError(f"Harris blockSize must be >= 2, got {params.block_size}")
    if params.ksize < 3 or params.ksize % 2 == 0:
        raise ValueError(f"Harris ksize must be an odd integer >= 3, got {params.ksize}")

    response = cv2.cornerHarris(gray, params.block_size, params.ksize, params.k)
    threshold = params.threshold_ratio * float(response.max())

    # `or` would treat a deliberate distance of 0 as "missing" and replace it.
    if params.nms_distance is None:
        min_distance = max(5, int(min(h, w) * 0.01))
    else:
        min_distance = int(params.nms_distance)
        if min_distance < 0:
            raise ValueError(f"nms_distance must be >= 0, got {min_distance}")
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

    # Region finding (contour) -> match Harris maxima to its vertices.
    # A failed match does not give up: the same rule is retried on candidates
    # inside the region box, then on every candidate.
    region_vertices = _find_region_vertices(region_map) if region_map is not None else None

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
                corners = _refine_corners(refine_image, matched)
                method = "contour+harris"
            except CornerNotFoundError as exc:
                message = str(exc)
        else:
            message = "no Harris corner found near a grid vertex"

        if corners is None:
            inside = _candidates_in_bbox(candidates, region_vertices, gray.shape)
            fallback, fallback_message = _corners_from_candidates(inside, gray.shape, refine_image)
            if fallback is not None:
                corners = fallback
                method = "region-bbox+harris"
                message = ""
            elif fallback_message:
                message = fallback_message

    if corners is None:
        fallback, fallback_message = _corners_from_candidates(candidates, gray.shape, refine_image)
        if fallback is not None:
            corners = fallback
            method = "harris-rule"
            message = ""
        elif not message:
            message = fallback_message

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
