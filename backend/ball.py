import math

import numpy as np
from ultralytics import YOLO

# COCO class id for sports ball
BALL_CLASS_ID = 32
DEFAULT_BALL_MODEL = "yolov8s.pt"


class BallDetector:
    """Detects football with YOLOv8 using spatial tracking for consistency."""

    def __init__(
        self,
        model_path: str = DEFAULT_BALL_MODEL,
        conf_threshold: float = 0.15,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.last_center: tuple[int, int] | None = None

    def detect(
        self, frame: np.ndarray, device: str | None = None
    ) -> tuple[list[int] | None, tuple[int, int] | None]:
        """Detects the ball in the frame, returning (bbox, center) or (None, None)."""
        dev = device or self.device
        results = self.model.predict(
            frame,
            classes=[BALL_CLASS_ID],
            conf=self.conf_threshold,
            device=dev,
            verbose=False,
        )

        candidates: list[tuple[list[int], tuple[int, int], float]] = []
        for res in results:
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    candidates.append(([x1, y1, x2, y2], center, conf))

        if candidates:
            best_bbox, best_center = self._select_best_candidate(candidates)
            self.last_center = best_center
            return best_bbox, best_center

        self.last_center = None
        return None, None

    def _select_best_candidate(
        self, candidates: list[tuple[list[int], tuple[int, int], float]]
    ) -> tuple[list[int], tuple[int, int]]:
        """Picks the candidate closest to the previous known ball position."""
        if self.last_center is None or len(candidates) == 1:
            best = max(candidates, key=lambda c: c[2])
            return best[0], best[1]

        def score(c: tuple[list[int], tuple[int, int], float]) -> float:
            dist = math.hypot(
                c[1][0] - self.last_center[0], c[1][1] - self.last_center[1]
            )
            distance_penalty = max(0.0, 1.0 - (dist / 250.0))
            return c[2] * 0.4 + distance_penalty * 0.6

        best = max(candidates, key=score)
        return best[0], best[1]

    def reset(self) -> None:
        """Resets tracking history."""
        self.last_center = None
