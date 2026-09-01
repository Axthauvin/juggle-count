from collections import deque

import cv2
import numpy as np


def draw_visualizations(
    frame: np.ndarray,
    keypoints: list[tuple[int, int]],
    ball_bbox: list[int] | None,
    ball_center: tuple[int, int] | None,
    ball_trail: deque[tuple[int, int]],
    juggle_count: int,
    state: str,
    bounced: bool,
) -> np.ndarray:
    """Draws keypoints, ball bounding box, trajectory trail, and HUD onto frame."""
    out = frame.copy()

    # Draw body joints
    for kx, ky in keypoints:
        cv2.circle(out, (kx, ky), 5, (0, 255, 255), -1)

    # Draw ball trajectory trail
    for i in range(1, len(ball_trail)):
        cv2.line(out, ball_trail[i - 1], ball_trail[i], (0, 165, 255), 2)

    # Draw ball bounding box and center
    if ball_bbox is not None:
        bx1, by1, bx2, by2 = ball_bbox
        box_color = (0, 255, 0) if bounced else (0, 215, 255)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), box_color, 2)
        if ball_center is not None:
            cv2.circle(out, ball_center, 4, (0, 0, 255), -1)

    # Draw semi-transparent HUD overlay
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (230, 85), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    cv2.putText(
        out,
        f"Juggles: {juggle_count}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    state_color = (0, 200, 255) if state == "FALLING" else (255, 150, 50)
    cv2.putText(
        out,
        f"State: {state}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )

    return out
