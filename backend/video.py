import logging
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import imageio
import imageio_ffmpeg
import numpy as np

from backend.ball import BallDetector
from backend.counter import JuggleCounter
from backend.pose import PoseDetector
from backend.visualizer import draw_visualizations

logger = logging.getLogger("uvicorn.info")


def cleanup_file(path: str | None) -> None:
    """Removes file if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def get_video_rotation(filepath: str) -> int:
    """Detects video clockwise rotation metadata (0, 90, 180, 270) using OpenCV/FFmpeg."""

    # This is quite long to explain all the issues that i had here,
    # when i take a video with my iPhone, it has a rotation metadata of 90 degrees,
    # but when i read it with OpenCV, it returns 0 degrees.
    # This function purposely tries multiple methods to get the correct rotation, but it may not be perfect for all videos.

    try:
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            cap.release()
            if meta in (90, 180, 270):
                return int(meta)
    except Exception:  # noqa: BLE001, S110 ruff ?
        pass

    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [exe, "-i", filepath],
            capture_output=True,
            text=True,
            errors="ignore",
            check=False,
        )
        stderr = res.stderr

        # Check displaymatrix rotation
        match_matrix = re.search(
            r"displaymatrix:\s*rotation of\s*([-\d.]+)\s*degrees",
            stderr,
            re.IGNORECASE,
        )
        if match_matrix:
            return round(-float(match_matrix.group(1))) % 360

        # Check rotate tag
        match_rotate = re.search(r"rotate\s*:\s*(\d+)", stderr, re.IGNORECASE)
        if match_rotate:
            return int(match_rotate.group(1)) % 360

        # Check rotation side data
        match_side = re.search(r"rotation:\s*([-\d.]+)", stderr, re.IGNORECASE)
        if match_side:
            return round(-float(match_side.group(1))) % 360
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Could not determine video rotation: %s", exc)

    return 0


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Applies clockwise rotation (90, 180, or 270 degrees) to frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def process_video_file(
    input_path: str,
    output_path: str | None,
    ball_detector: BallDetector,
    pose_detector: PoseDetector | None = None,
    filename: str = "video.mp4",
    device: str = "cpu",
    annotate: bool = True,
    jobs: dict | None = None,
    job_id: str | None = None,
) -> dict:
    """Processes video, counts juggles, logs progress, and saves annotated MP4."""
    rotation = get_video_rotation(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        if jobs is not None and job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Failed to open source video file."
        raise ValueError("Failed to open source video file.")

    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except (cv2.error, OSError) as exc:
        logger.debug("Could not disable auto orientation: %s", exc)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    est_duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

    display_name = Path(filename).name
    if rotation != 0:
        logger.info("Video '%s': applying %d deg rotation", display_name, rotation)

    logger.info(
        "Processing '%s' on %s (%s frames, %.1f FPS, ~%.1fs)",
        display_name,
        device.upper(),
        total_frames if total_frames > 0 else "?",
        fps,
        est_duration,
    )

    writer = None
    if annotate and output_path:
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",  # type: ignore (ruff is crazy)
            pixelformat="yuv420p",
            macro_block_size=None,
        )

    detector_pose = pose_detector or PoseDetector(device=device)
    counter = JuggleCounter()
    events: list[dict] = []
    frame_idx = 0

    start_time = time.time()
    last_log_time = start_time
    log_interval = 1.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation != 0:
                frame = apply_rotation(frame, rotation)

            keypoints = detector_pose.detect(frame, device=device)
            ball_bbox, ball_center = ball_detector.detect(frame, device=device)
            count, state, bounced = counter.update(ball_center, keypoints)

            if bounced:
                timestamp = round(frame_idx / fps, 2)
                events.append(
                    {
                        "juggle_number": count,
                        "frame": frame_idx,
                        "timestamp_seconds": timestamp,
                    }
                )
                logger.info(
                    "Bounce detected on '%s'! Juggle #%d at %.2fs (frame %d)",
                    display_name,
                    count,
                    timestamp,
                    frame_idx,
                )

            if writer is not None:
                annotated_frame = draw_visualizations(
                    frame=frame,
                    keypoints=keypoints,
                    ball_bbox=ball_bbox,
                    ball_center=ball_center,
                    ball_trail=counter.trail,
                    juggle_count=count,
                    state=state,
                    bounced=bounced,
                )
                # Convert BGR (OpenCV) to RGB (imageio)
                rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_annotated)

            frame_idx += 1

            # Update job progress percentage (0 - 100)
            if jobs is not None and job_id in jobs and total_frames > 0:
                jobs[job_id]["progress"] = round((frame_idx / total_frames) * 100, 1)

            # Log periodic progress
            now = time.time()
            if now - last_log_time >= log_interval:
                elapsed = now - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                if total_frames > 0:
                    pct = (frame_idx / total_frames) * 100
                    logger.info(
                        "Progress '%s': %3.0f%% (%d/%d frames) | %.1f FPS | Juggles: %d",
                        display_name,
                        pct,
                        frame_idx,
                        total_frames,
                        current_fps,
                        counter.count,
                    )
                else:
                    logger.info(
                        "Progress '%s': %d frames | %.1f FPS | Juggles: %d",
                        display_name,
                        frame_idx,
                        current_fps,
                        counter.count,
                    )
                last_log_time = now

    except Exception as exc:
        logger.error("Error while processing video '%s': %s", display_name, exc)
        if jobs is not None and job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        cleanup_file(input_path)

    total_elapsed = time.time() - start_time
    avg_fps = (frame_idx / total_elapsed) if total_elapsed > 0 else 0.0
    duration = round(frame_idx / fps, 2) if fps > 0 else 0.0

    stats = {
        "total_frames": frame_idx,
        "fps": round(fps, 2),
        "duration_seconds": duration,
        "juggle_count": counter.count,
        "events": events,
    }

    if jobs is not None and job_id in jobs:
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["stats"] = stats

    logger.info(
        "Finished '%s' in %.2fs (%.1f FPS) | Total juggles: %d",
        display_name,
        total_elapsed,
        avg_fps,
        counter.count,
    )

    return stats
