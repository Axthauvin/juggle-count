// API base URL
const API_BASE = window.location.origin.startsWith("http")
  ? window.location.origin
  : "http://localhost:8080";

const isFrench = document.documentElement.lang === "fr";

let isGpuAvailable = false;
let useGpu = false;
let videoEvents = [];
let prevLiveCount = -1;

const dropzone = document.getElementById("dropzone");
const uploadView = document.getElementById("upload-view");
const loadingState = document.getElementById("loading-state");
const resultSection = document.getElementById("result-section");
const annotatedVideo = document.getElementById("annotated-video");
const downloadBtn = document.getElementById("download-btn");
const liveCountElem = document.getElementById("res-live-count");

// Check GPU availability on startup
async function initDeviceCapabilities() {
  const deviceDot = document.getElementById("device-dot");
  const statusText = document.getElementById("device-status-text");
  const gpuToggle = document.getElementById("gpu-toggle");

  try {
    const res = await fetch(`${API_BASE}/device-info`);
    const info = await res.json();
    isGpuAvailable = info.gpu_available;

    if (isGpuAvailable) {
      const saved = localStorage.getItem("juggle_use_gpu");
      useGpu = saved !== null ? saved === "true" : true;
      gpuToggle.checked = useGpu;
      gpuToggle.disabled = false;
    } else {
      isGpuAvailable = false;
      useGpu = false;
      gpuToggle.checked = false;
      gpuToggle.disabled = true;
    }
  } catch {
    isGpuAvailable = false;
    useGpu = false;
    gpuToggle.disabled = true;
  }
  updateDeviceUI(deviceDot, statusText);
}

function onGpuToggleChanged(e) {
  useGpu = e.target.checked && isGpuAvailable;
  localStorage.setItem("juggle_use_gpu", useGpu);
  updateDeviceUI(
    document.getElementById("device-dot"),
    document.getElementById("device-status-text")
  );
}

function updateDeviceUI(dot, text) {
  if (!dot || !text) return;
  if (useGpu) {
    dot.className = "device-dot gpu-active";
    text.innerText = "GPU (CUDA)";
  } else {
    dot.className = "device-dot cpu-active";
    text.innerText = isFrench
      ? (isGpuAvailable ? "CPU (GPU désactivé)" : "CPU (Par défaut)")
      : (isGpuAvailable ? "CPU (GPU disabled)" : "CPU (Default)");
  }
}

// Synchronize juggle count indicator with current video playback timestamp
function updateLiveCount() {
  if (!annotatedVideo) return;
  const t = annotatedVideo.currentTime || 0;

  let count = 0;
  for (const ev of videoEvents) {
    if (ev.timestamp_seconds <= t + 0.05) {
      count = ev.juggle_number ?? count + 1;
    } else {
      break;
    }
  }

  if (liveCountElem && count !== prevLiveCount) {
    liveCountElem.innerText = count;
    if (count > prevLiveCount && prevLiveCount !== -1) {
      liveCountElem.classList.add("bump");
      setTimeout(() => liveCountElem.classList.remove("bump"), 150);
    }
    prevLiveCount = count;
  }
}

function onVideoPlaybackFrame() {
  updateLiveCount();
  if (annotatedVideo && !annotatedVideo.paused && !annotatedVideo.ended) {
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      annotatedVideo.requestVideoFrameCallback(onVideoPlaybackFrame);
    } else {
      requestAnimationFrame(onVideoPlaybackFrame);
    }
  }
}

annotatedVideo.addEventListener("play", onVideoPlaybackFrame);
annotatedVideo.addEventListener("timeupdate", updateLiveCount);
annotatedVideo.addEventListener("seeked", updateLiveCount);
annotatedVideo.addEventListener("seeking", updateLiveCount);

// Drag-and-drop event listeners
["dragenter", "dragover"].forEach((evt) => {
  dropzone?.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone?.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone?.addEventListener("drop", (e) => {
  if (e.dataTransfer.files?.[0]) {
    uploadVideoFile(e.dataTransfer.files[0]);
  }
});

function handleFileSelected(e) {
  if (e.target.files?.[0]) {
    uploadVideoFile(e.target.files[0]);
  }
}

// Load a sample video from assets and process
async function loadSampleVideo(url, filename) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(isFrench ? "Impossible de charger la vidéo d'exemple" : "Could not load sample video");
    const blob = await res.blob();
    const file = new File([blob], filename, { type: "video/mp4" });
    uploadVideoFile(file);
  } catch (err) {
    alert((isFrench ? "Erreur : " : "Error: ") + err.message);
  }
}

// Upload video and poll progress
async function uploadVideoFile(file) {
  if (uploadView) uploadView.style.display = "none";
  else if (dropzone) dropzone.style.display = "none";

  loadingState.style.display = "block";
  resultSection.style.display = "none";

  const progressBar = document.getElementById("progress-bar");
  const progressPercent = document.getElementById("progress-percent");
  progressBar.style.width = "0%";
  progressPercent.innerText = "0%";

  document.getElementById("loading-engine-text").innerText = useGpu
    ? (isFrench ? "Inférence en cours sur GPU (CUDA)..." : "Inference running on GPU (CUDA)...")
    : (isFrench ? "Inférence en cours sur CPU..." : "Inference running on CPU...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/process-video?use_gpu=${useGpu}`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || (isFrench ? "Erreur pendant l'envoi." : "Error during upload."));
    }

    const { job_id } = await res.json();
    const jobData = await pollJobProgress(job_id, progressBar, progressPercent);

    const stats = jobData.stats || {};
    videoEvents = stats.events || [];
    prevLiveCount = 0;
    if (liveCountElem) liveCountElem.innerText = "0";

    document.getElementById("res-juggle-count").innerText =
      stats.juggle_count ?? 0;
    document.getElementById("res-duration").innerText =
      `${stats.duration_seconds ?? 0}s`;
    document.getElementById("res-frames").innerText = stats.total_frames ?? 0;

    const resultUrl = `${API_BASE}/process-video/${job_id}/result`;
    annotatedVideo.src = resultUrl;
    annotatedVideo.load();
    downloadBtn.href = resultUrl;

    loadingState.style.display = "none";
    resultSection.style.display = "block";
  } catch (err) {
    alert((isFrench ? "Erreur : " : "Error: ") + err.message);
    resetUpload();
  }
}

// Poll job progress from backend
function pollJobProgress(jobId, progressBar, progressPercent) {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/process-video/${jobId}/progress`);
        if (!res.ok) {
          clearInterval(interval);
          return reject(new Error(isFrench ? "Échec de récupération de la progression" : "Failed to fetch job progress"));
        }

        const data = await res.json();
        const pct = Math.min(100, Math.max(0, data.progress || 0));
        progressBar.style.width = `${pct}%`;
        progressPercent.innerText = `${Math.round(pct)}%`;

        if (data.status === "completed") {
          clearInterval(interval);
          resolve(data);
        } else if (data.status === "failed") {
          clearInterval(interval);
          reject(new Error(data.error || (isFrench ? "Échec du traitement vidéo" : "Video processing failed")));
        }
      } catch (err) {
        clearInterval(interval);
        reject(err);
      }
    }, 250);
  });
}

// Reset upload form
function resetUpload() {
  const videoInput = document.getElementById("video-input");
  if (videoInput) videoInput.value = "";
  if (uploadView) uploadView.style.display = "block";
  else if (dropzone) dropzone.style.display = "block";

  loadingState.style.display = "none";
  resultSection.style.display = "none";
  videoEvents = [];
  prevLiveCount = -1;
  if (liveCountElem) liveCountElem.innerText = "0";
  annotatedVideo.pause();
  annotatedVideo.src = "";
}

// Initialize on page load
initDeviceCapabilities();
