"""Depth geometry and lightweight tracking for the three known Task E objects."""

from dataclasses import dataclass

import numpy as np

from .camera import FixedCamera
from .constants import (
    BOTTLE_BODY_LOCALIZE_MIN_POINTS, BOTTLE_BODY_LOCALIZE_Z_MAX,
    BOTTLE_BODY_LOCALIZE_Z_MIN, OBJECT_Y_BANDS, TABLE_TOP_Z,
)


@dataclass
class ObjectEstimate:
    object_id: int
    xy: np.ndarray
    long_axis_xy: np.ndarray
    aspect_ratio: float
    point_count: int
    mean_rgb: np.ndarray
    body_y_point_count: int = 0
    body_y_shift_m: float = 0.0
    banana_raw_point_count: int = 0
    banana_mid_point_count: int = 0
    banana_span_m: float = 0.0
    banana_shift_x_m: float = 0.0
    banana_shift_y_m: float = 0.0


class TaskPerception:
    def __init__(self, width=640, height=480):
        self.camera = FixedCamera(width, height)
        self.points = None
        self.rgb_samples = None

    def update(self, rgb, depth):
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
            rgb = np.moveaxis(rgb, 0, -1)
        if rgb.ndim != 3 or rgb.shape[:2] != (self.camera.height, self.camera.width):
            raise ValueError(f"video_rgb has unexpected shape {rgb.shape}")
        self.points, rows, cols = self.camera.point_cloud(depth)
        self.rgb_samples = rgb[rows, cols, :3]

    def _object_points(self, object_id, xy_hint=None):
        if self.points is None:
            return np.empty((0, 3)), np.empty((0, 3))
        p = self.points
        low_y, high_y = OBJECT_Y_BANDS[object_id]
        mask = (
            (p[:, 0] > 0.82) & (p[:, 0] < 1.18)
            & (p[:, 1] > low_y - 0.035) & (p[:, 1] < high_y + 0.035)
            & (p[:, 2] > TABLE_TOP_Z + 0.012)
            & (p[:, 2] < TABLE_TOP_Z + (0.24 if object_id == 2 else 0.15))
        )
        if xy_hint is not None:
            mask &= (np.abs(p[:, 0] - xy_hint[0]) < 0.09)
            mask &= (np.abs(p[:, 1] - xy_hint[1]) < 0.08)
        return p[mask], self.rgb_samples[mask]

    @staticmethod
    def _banana_component(points, colors):
        """Exclude disconnected neighbouring objects before locating the banana."""
        if len(points) < 24:
            return points, colors
        xy = points[:, :2]
        # Include height: the bottle's lower body can overlap the banana's
        # broad XY crop without belonging to the same physical surface.
        neighbors = np.sum((points[:, None, :] - points[None, :, :]) ** 2,
                           axis=2) < 0.018 ** 2
        unseen = set(range(len(points)))
        components = []
        while unseen:
            seed = unseen.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                adjacent = set(np.flatnonzero(neighbors[frontier.pop()])) & unseen
                unseen.difference_update(adjacent)
                component.update(adjacent)
                frontier.extend(adjacent)
            components.append(np.fromiter(component, dtype=int))
        eligible = [indices for indices in components
                    if len(indices) >= max(12, int(0.55 * len(points)))
                    and 0.015 <= np.median(xy[indices, 1]) <= 0.105]
        if not eligible:
            return points, colors
        chosen = max(eligible, key=len)
        return points[chosen], colors[chosen]

    def detect(self, object_id):
        points, colors = self._object_points(object_id)
        if len(points) < 12:
            return None
        banana_raw_point_count = len(points) if object_id == 3 else 0
        if object_id == 3:
            points, colors = self._banana_component(points, colors)
        xy = points[:, :2]
        # Trim depth edges and stray points before estimating the object centre.
        lo = np.quantile(xy, 0.08, axis=0)
        hi = np.quantile(xy, 0.92, axis=0)
        trimmed = xy[np.all((xy >= lo) & (xy <= hi), axis=1)]
        if len(trimmed) < 8:
            return None
        center = np.median(trimmed, axis=0)
        body_y_point_count = 0
        body_y_shift_m = 0.0
        if object_id == 2:
            # The fixed Y crop may include the sugar box or banana when their
            # edges lie close to the bottle. Their surfaces are lower than
            # the upright bottle's upper body. Change only Y: the calibrated
            # X correction depends on the original front-face median.
            body = points[(points[:, 2] >= BOTTLE_BODY_LOCALIZE_Z_MIN)
                          & (points[:, 2] <= BOTTLE_BODY_LOCALIZE_Z_MAX)]
            if len(body) >= BOTTLE_BODY_LOCALIZE_MIN_POINTS:
                body_y = float(np.median(body[:, 1]))
                body_y_shift_m = body_y - float(center[1])
                center[1] = body_y
                body_y_point_count = len(body)
        covariance = np.cov(trimmed.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        long_axis = eigenvectors[:, np.argmax(eigenvalues)]
        if long_axis[0] < 0:
            long_axis = -long_axis
        banana_mid_point_count = 0
        banana_span_m = 0.0
        banana_shift = np.zeros(2)
        if object_id == 3:
            # A curved banana can contribute many visible points near one
            # end, pulling the point median toward its head. Place the jaws
            # at the middle of its robust length, and use only the middle
            # section to centre across the banana. Keep the original centre
            # when the cloud is too short, sparse, or distorted by clutter.
            # Use the near-end quantiles of the connected component, not a
            # second trim of an already trimmed cloud: the latter can move
            # the estimated middle back toward a densely sampled banana tip.
            along = xy @ long_axis
            start, end = np.quantile(along, (0.02, 0.98))
            span = float(end - start)
            if 0.06 <= span <= 0.25:
                middle_along = 0.5 * (start + end)
                middle_mask = np.abs(along - middle_along) <= 0.20 * span
                middle_points = xy[middle_mask]
                if len(middle_points) >= max(12, int(0.12 * len(xy))):
                    across_axis = np.array([-long_axis[1], long_axis[0]])
                    middle_across = float(np.median(middle_points @ across_axis))
                    middle_xy = (middle_along * long_axis
                                 + middle_across * across_axis)
                    shift = middle_xy - center
                    nearest_point = np.min(np.linalg.norm(xy - middle_xy,
                                                          axis=1))
                    if np.linalg.norm(shift) <= 0.06 and nearest_point <= 0.025:
                        center = middle_xy
                        banana_mid_point_count = len(middle_points)
                        banana_span_m = span
                        banana_shift = shift
        return ObjectEstimate(
            object_id=object_id,
            xy=center,
            long_axis_xy=long_axis,
            aspect_ratio=float(np.sqrt(max(eigenvalues) / max(min(eigenvalues), 1e-9))),
            point_count=len(points),
            mean_rgb=colors.mean(axis=0),
            body_y_point_count=body_y_point_count,
            body_y_shift_m=body_y_shift_m,
            banana_raw_point_count=banana_raw_point_count,
            banana_mid_point_count=banana_mid_point_count,
            banana_span_m=banana_span_m,
            banana_shift_x_m=float(banana_shift[0]),
            banana_shift_y_m=float(banana_shift[1]),
        )

    def table_count(self, object_id, xy_hint):
        points, _ = self._object_points(object_id, xy_hint)
        return len(points)

    def basket_count(self, slot_xy):
        if self.points is None:
            return 0
        p = self.points
        mask = (
            (np.abs(p[:, 0] - slot_xy[0]) < 0.07)
            & (np.abs(p[:, 1] - slot_xy[1]) < 0.07)
            & (p[:, 2] > TABLE_TOP_Z + 0.015)
            & (p[:, 2] < TABLE_TOP_Z + 0.19)
        )
        return int(mask.sum())
