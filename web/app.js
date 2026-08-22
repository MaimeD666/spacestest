const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#video-input");
const dropZone = document.querySelector("#drop-zone");
const fileRow = document.querySelector("#file-row");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const submitButton = document.querySelector("#submit-button");
const processPanel = document.querySelector("#process-panel");
const statusChip = document.querySelector("#status-chip");
const stageText = document.querySelector("#stage-text");
const progressValue = document.querySelector("#progress-value");
const progressTrack = document.querySelector("#progress-track");
const progressFill = document.querySelector("#progress-fill");
const jobIdElement = document.querySelector("#job-id");
const errorBox = document.querySelector("#error-box");
const results = document.querySelector("#results");
const characterGrid = document.querySelector("#character-grid");
const newJobButton = document.querySelector("#new-job-button");

let selectedFile = null;
let activeJobId = null;
let pollTimer = null;

const statusLabels = {
  queued: "QUEUED",
  analyzing: "ANALYSIS",
  deduplicating: "DEDUP",
  generating: "GENERATE",
  completed: "DONE",
  failed: "FAILED",
};

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function setSelectedFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".mp4")) {
    showClientError("Нужен файл в формате MP4");
    return;
  }
  if (file.size > 100 * 1024 * 1024) {
    showClientError("Файл больше 100 МБ");
    return;
  }

  selectedFile = file;
  document.body.classList.add("has-file");
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileRow.hidden = false;
  submitButton.disabled = false;
  errorBox.hidden = true;
}

function showClientError(message) {
  document.body.classList.add("has-job");
  processPanel.hidden = false;
  errorBox.textContent = message;
  errorBox.hidden = false;
  setProgress(0, "Ошибка файла", "failed");
  processPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function setProgress(percent, stage, status) {
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  progressFill.style.width = `${safePercent}%`;
  progressValue.textContent = `${safePercent}%`;
  progressTrack.setAttribute("aria-valuenow", String(safePercent));
  stageText.textContent = stage;
  statusChip.textContent = statusLabels[status] || String(status).toUpperCase();
}

function parseError(request) {
  try {
    const payload = JSON.parse(request.responseText);
    if (typeof payload.detail === "string") return payload.detail;
  } catch (_error) {
    // The fallback below is deliberately plain: server responses can be empty.
  }
  return `Ошибка сервера (${request.status || "нет соединения"})`;
}

function uploadVideo(file) {
  const data = new FormData();
  data.append("video", file);
  const request = new XMLHttpRequest();
  request.open("POST", "/v1/jobs");

  document.body.classList.add("has-job");
  processPanel.hidden = false;
  results.hidden = true;
  characterGrid.replaceChildren();
  errorBox.hidden = true;
  submitButton.disabled = true;
  jobIdElement.textContent = "создаётся";
  setProgress(1, "Загрузка видео", "queued");
  processPanel.scrollIntoView({ behavior: "smooth", block: "center" });

  request.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    setProgress((event.loaded / event.total) * 10, "Загрузка видео", "queued");
  });

  request.addEventListener("load", () => {
    if (request.status !== 202) {
      failUpload(parseError(request));
      return;
    }
    const job = JSON.parse(request.responseText);
    activeJobId = job.job_id;
    jobIdElement.textContent = activeJobId;
    window.history.replaceState(null, "", `/?job=${activeJobId}`);
    renderJob(job);
    schedulePoll(500);
  });

  request.addEventListener("error", () => {
    failUpload("Не удалось соединиться с локальным сервером");
  });

  request.send(data);
}

function failUpload(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  submitButton.disabled = false;
  setProgress(0, "Загрузка не выполнена", "failed");
}

function renderJob(job) {
  const overallProgress = job.status === "completed" ? 100 : 10 + job.progress * 0.9;
  setProgress(overallProgress, job.stage, job.status);
  jobIdElement.textContent = job.job_id;

  if (job.status === "failed") {
    errorBox.textContent = job.error || "Обработка завершилась с ошибкой";
    errorBox.hidden = false;
    submitButton.disabled = false;
    return;
  }

  if (job.status === "completed") {
    errorBox.hidden = true;
    renderCharacters(job.characters);
    submitButton.disabled = false;
  }
}

function renderCharacters(characters) {
  characterGrid.replaceChildren();
  for (const character of characters) {
    const card = document.createElement("article");
    card.className = "character-card";

    const imageWrap = document.createElement("div");
    imageWrap.className = "character-image-wrap";
    const image = document.createElement("img");
    image.className = "character-image";
    image.src = character.image_url;
    image.alt = `Ростовое изображение ${character.character_id}`;
    image.loading = "lazy";
    imageWrap.append(image);

    const meta = document.createElement("div");
    meta.className = "character-meta";
    const title = document.createElement("h3");
    title.textContent = character.character_id.replace("_", " ");
    const facts = document.createElement("div");
    facts.className = "character-facts";
    const confidence = document.createElement("span");
    confidence.textContent = `CONF ${Math.round(character.confidence * 100)}%`;
    const tracks = document.createElement("span");
    tracks.textContent = `TRACK ${character.track_ids.join("+")}`;
    const style = document.createElement("span");
    style.textContent = String(character.media_style || "auto")
      .replace("_animation", "")
      .toUpperCase();
    facts.append(confidence, style, tracks);

    const download = document.createElement("a");
    download.className = "download-link";
    download.href = character.image_url;
    download.download = `${character.character_id}.png`;
    download.textContent = "СКАЧАТЬ PNG ↓";

    meta.append(title, facts, download);
    card.append(imageWrap, meta);
    characterGrid.append(card);
  }
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function pollJob() {
  if (!activeJobId) return;
  try {
    const response = await fetch(`/v1/jobs/${activeJobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const job = await response.json();
    renderJob(job);
    if (job.status !== "completed" && job.status !== "failed") {
      schedulePoll(1200);
    }
  } catch (_error) {
    setProgress(
      Number.parseInt(progressValue.textContent, 10) || 10,
      "Связь потеряна — повторяем запрос",
      "queued",
    );
    schedulePoll(2500);
  }
}

function schedulePoll(delay) {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(pollJob, delay);
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  setSelectedFile(file);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (selectedFile) uploadVideo(selectedFile);
});

newJobButton.addEventListener("click", () => {
  window.clearTimeout(pollTimer);
  activeJobId = null;
  selectedFile = null;
  form.reset();
  fileRow.hidden = true;
  processPanel.hidden = true;
  results.hidden = true;
  submitButton.disabled = true;
  characterGrid.replaceChildren();
  document.body.classList.remove("has-job");
  document.body.classList.remove("has-file");
  window.history.replaceState(null, "", "/");
  dropZone.scrollIntoView({ behavior: "smooth", block: "center" });
});

const jobFromUrl = new URLSearchParams(window.location.search).get("job");
if (jobFromUrl && /^[a-f0-9]{32}$/.test(jobFromUrl)) {
  document.body.classList.add("has-job");
  activeJobId = jobFromUrl;
  processPanel.hidden = false;
  jobIdElement.textContent = activeJobId;
  setProgress(10, "Восстанавливаем состояние", "queued");
  pollJob();
}
