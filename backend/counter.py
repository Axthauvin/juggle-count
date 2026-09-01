import math
from collections import deque

DEFAULT_PROXIMITY_THRESHOLD = 130
TRAIL_MAXLEN = 20
MAX_MISSED_FRAMES = 10


class JuggleCounter:
    """Tracks ball trajectory and counts valid juggle bounces."""

    def __init__(self, proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD):
        self.proximity_threshold = proximity_threshold
        self.trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_MAXLEN)
        self.count: int = 0
        self.state: str = "FALLING"
        self.consecutive_misses: int = 0

    def update(
        self,
        ball_center: tuple[int, int] | None,
        keypoints: list[tuple[int, int]],
    ) -> tuple[int, str, bool]:
        """Updates trajectory and returns (count, state, bounced)."""
        bounced = False

        if ball_center is not None:
            self.consecutive_misses = 0
            self.trail.append(ball_center)
            _bx, by = ball_center

            if len(self.trail) >= 4:
                prev_y = self.trail[-3][1]
                dy = by - prev_y

                # Ball moving downwards
                if dy > 3:
                    self.state = "FALLING"

                # Ball reversing upwards near player body
                elif dy < -2 and self.state == "FALLING":
                    near_body = False
                    if keypoints:
                        recent_pts = list(self.trail)[-3:]
                        min_dist = min(
                            math.hypot(px - kx, py - ky)
                            for px, py in recent_pts
                            for kx, ky in keypoints
                        )
                        if min_dist < self.proximity_threshold:
                            near_body = True

                    if near_body:
                        self.count += 1
                        self.state = "RISING"
                        bounced = True
        else:
            self.consecutive_misses += 1
            if self.consecutive_misses > MAX_MISSED_FRAMES:
                self.trail.clear()

        return self.count, self.state, bounced
