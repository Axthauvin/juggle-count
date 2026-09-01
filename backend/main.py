import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.ball import BallDetector
from backend.pose import PoseDetector
from backend.video import process_video_file

app = FastAPI(
    title="Juggle Count API",
    description="Video analysis API for counting football juggles with YOLOv8.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ball_detector = BallDetector()
pose_detector = PoseDetector()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

jobs: dict[str, dict] = {}

logger = logging.getLogger("uvicorn.info")


def get_system_device_info() -> dict:
    """Returns GPU availability and device name."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "default_device": "cuda" if gpu_available else "cpu",
    }


def resolve_device(use_gpu: bool) -> str:
    """Returns 'cuda' if GPU is requested and available, else 'cpu'."""
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"


@app.get("/")
async def serve_frontend():
    """Serves frontend index.html."""
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return {
        "message": "Juggle Count API is running. Access /docs for API documentation."
    }


@app.get("/device-info")
async def device_info():
    """Returns system device capabilities."""
    info = get_system_device_info()
    logger.info(
        "Device info: GPU Available=%s, GPU Name=%s, Default=%s",
        info["gpu_available"],
        info["gpu_name"],
        info["default_device"],
    )
    return info


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Video file to analyze")],
    use_gpu: Annotated[bool, Query()] = False,
):
    """Starts background video processing and returns job ID."""
    filename = file.filename or "video.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: .mp4, .mov, .avi, .webm, .mkv",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_in:
        input_path = temp_in.name
        temp_in.write(await file.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
        output_path = temp_out.name

    device = resolve_device(use_gpu)
    job_id = str(uuid4())

    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "error": None,
        "stats": None,
        "output_path": output_path,
        "filename": filename,
        "device": device,
    }

    background_tasks.add_task(
        process_video_file,
        input_path=input_path,
        output_path=output_path,
        ball_detector=ball_detector,
        pose_detector=pose_detector,
        filename=filename,
        device=device,
        annotate=True,
        jobs=jobs,
        job_id=job_id,
    )

    return JSONResponse(
        content={"job_id": job_id, "status": "processing", "device": device}
    )


@app.get("/process-video/{job_id}/progress")
async def get_progress(job_id: str):
    """Returns status, progress percentage (0-100), and stats when completed."""
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "stats": job["stats"],
    }


@app.get("/process-video/{job_id}/result")
async def get_result(job_id: str):
    """Streams the processed annotated MP4 video file."""
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"annotated_{Path(job['filename']).stem}.mp4",
    )


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def dev():
    """Starts local development server with auto-reload."""
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    dev()
