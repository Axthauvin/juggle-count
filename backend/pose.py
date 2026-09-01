import numpy as np
from ultralytics import YOLO

# Key COCO pose joint indices (nose, shoulders, hips, knees, ankles)
TARGET_JOINTS = [0, 5, 6, 11, 12, 13, 14, 15, 16]
DEFAULT_POSE_MODEL = "yolov8n-pose.pt"


class PoseDetector:
    """Detects human body keypoints using YOLOv8-Pose."""

    def __init__(
        self,
        model_path: str = DEFAULT_POSE_MODEL,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(
        self, frame: np.ndarray, device: str | None = None
    ) -> list[tuple[int, int]]:
        """Extracts pixel coordinates (x, y) of detected target joints."""
        dev = device or self.device
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=dev,
            verbose=False,
        )

        keypoints: list[tuple[int, int]] = []
        for res in results:
            if res.keypoints is not None and len(res.keypoints.xy) > 0:
                xy_persons = res.keypoints.xy
                conf_persons = res.keypoints.conf

                for p_idx, person_pts in enumerate(xy_persons):
                    for lm_id in TARGET_JOINTS:
                        if lm_id < len(person_pts):
                            kx, ky = (
                                float(person_pts[lm_id][0]),
                                float(person_pts[lm_id][1]),
                            )
                            if kx > 0 and ky > 0:
                                if (
                                    conf_persons is not None
                                    and float(conf_persons[p_idx][lm_id]) < 0.3
                                ):
                                    continue
                                keypoints.append((int(kx), int(ky)))

        return keypoints
