import { state } from "./state.js";
import { setupMarkdown } from "./utils.js";
import { checkAuthStatus, handleLogout } from "./auth.js";

// --- Studio State ---
const studioState = {
  model: "sdxl-lightning",
  mode: "text-to-image",
  ratio: "1:1",
  width: 1024,
  height: 1024,
  batchSize: 1,
  steps: 4,
  inputImage: null,
  hardware: null,
  lastResult: null,
  gallery: [],
};

// Initial DOM pointers
const _domMap = {
  sidebarToggle: "sidebar-toggle",
  sidebar: "sidebar",
  sidebarOverlay: "sidebar-overlay",
  logoutBtn: "logout-btn",
  currentUsernameSpan: "current-username",
  gpuName: "gpu-name",
  vramInfo: "vram-info",
  modelSelect: "studio-model",
  imgUploadGroup: "studio-img-upload",
  dropZone: "studio-drop-zone",
  fileInput: "studio-file-input",
  inputPreview: "studio-input-preview",
  prompt: "studio-prompt",
  negativePrompt: "studio-negative-prompt",
  resReadout: "studio-res-readout",
  batchInput: "studio-batch",
  batchVal: "studio-batch-val",
  stepsInput: "studio-steps",
  stepsVal: "studio-steps-val",
  generateBtn: "studio-generate-btn",
  timeEstimate: "studio-time-estimate",
  mainPreview: "studio-main-preview",
  placeholderState: "placeholder-state",
  resultImg: "studio-result-img",
  resultVideo: "studio-result-video",
  previewActions: "preview-actions",
  upscaleBtn: "studio-upscale-btn",
  downloadBtn: "studio-download-btn",
  loadingOverlay: "loading-overlay",
  loadingText: "studio-loading-text",
  gallery: "studio-gallery",
  sidebarGallery: "sidebar-gallery",
  lightbox: "lightbox",
  lightboxImg: "lightbox-img",
  lightboxDownload: "lightbox-download",
  lightboxClose: "lightbox-close",
  lightboxOverlay: "lightbox-overlay",
};

// Use Proxy for studio dom too for safety against missing IDs
const studioDom = new Proxy(
  {},
  {
    get: (target, prop) => {
      if (target[prop]) return target[prop];
      const id = _domMap[prop] || prop;
      let el = document.getElementById(id);
      if (!el) {
        // Try class search
        el = document.querySelector(id.startsWith(".") ? id : `.${id}`);
      }
      if (el) target[prop] = el; // cache it
      return el;
    },
  },
);

// For loopable elements like buttons
const getModeButtons = () => document.querySelectorAll(".mode-toggle button");
const getAspectButtons = () => document.querySelectorAll(".aspect-grid button");

// --- Initialization ---
async function init() {
  console.log("🚀 Bipod Studio Init v1.1 starting...");
  setupMarkdown();

  try {
    await checkAuthStatus();
    console.log("Auth check complete. state.currentUser:", state.currentUser);
  } catch (err) {
    console.error("Auth check failed:", err);
  }

  if (!state.authToken) {
    console.warn("No auth token, redirecting to home.");
    window.location.href = "/";
    return;
  }

  // Explicitly set username right away
  if (state.currentUser && studioDom.currentUsernameSpan) {
    console.log("Setting username to:", state.currentUser.username);
    studioDom.currentUsernameSpan.innerText = state.currentUser.username;
  } else {
    console.warn("User data or span missing:", state.currentUser);
  }

  setupEventListeners();
  await fetchHardwareStats();

  updateEstimates();
  console.log("✅ Studio Initialization complete.");
}

function setupEventListeners() {
  // Sidebar & Auth
  if (studioDom.sidebarToggle) {
    studioDom.sidebarToggle.addEventListener("click", () => {
      console.log("Sidebar toggle clicked");
      if (window.innerWidth <= 768) {
        studioDom.sidebar.classList.toggle("mobile-open");
        studioDom.sidebarOverlay.classList.toggle("active");
      } else {
        studioDom.sidebar.classList.toggle("hidden");
      }
    });
  }

  if (studioDom.sidebarOverlay) {
    studioDom.sidebarOverlay.addEventListener("click", () => {
      studioDom.sidebar.classList.remove("mobile-open");
      studioDom.sidebarOverlay.classList.remove("active");
    });
  }

  if (studioDom.logoutBtn) {
    studioDom.logoutBtn.addEventListener("click", handleLogout);
  }

  // Controls
  if (studioDom.modelSelect) {
    // Sync initial model
    studioState.model = studioDom.modelSelect.value || "sdxl-lightning";
    studioDom.modelSelect.addEventListener("change", (e) => {
      studioState.model = e.target.value;
      if (studioDom.stepsInput) {
        if (studioState.model === "sdxl-turbo") {
          studioDom.stepsInput.value = 1;
        } else if (studioState.model === "sdxl-lightning") {
          studioDom.stepsInput.value = 4;
        } else {
          studioDom.stepsInput.value = 30;
        }
        studioDom.stepsInput.dispatchEvent(new Event("input"));
      }
      updateEstimates();
    });
  }

  const modeButtons = getModeButtons();
  if (modeButtons) {
    modeButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        modeButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        studioState.mode = btn.dataset.mode;

        // Show upload group for both img2img and img2video
        if (studioDom.imgUploadGroup) {
          studioDom.imgUploadGroup.style.display =
            studioState.mode === "image-to-image" ||
            studioState.mode === "image-to-video"
              ? "block"
              : "none";
        }

        // Auto-switch model to SVD-XT if entering video mode
        if (studioState.mode === "image-to-video") {
          if (studioDom.modelSelect) {
            studioDom.modelSelect.value = "svd-xt";
            studioDom.modelSelect.dispatchEvent(new Event("input"));
          }
        }
      });
    });
  }

  if (studioDom.dropZone) {
    studioDom.dropZone.addEventListener("click", () =>
      studioDom.fileInput.click(),
    );
  }

  if (studioDom.fileInput) {
    studioDom.fileInput.addEventListener("change", (e) =>
      handleFileSelect(e.target.files[0]),
    );
  }

  if (studioDom.batchInput) {
    studioDom.batchInput.addEventListener("input", (e) => {
      studioState.batchSize = parseInt(e.target.value);
      if (studioDom.batchVal)
        studioDom.batchVal.innerText = studioState.batchSize;
      updateEstimates();
    });
  }

  if (studioDom.stepsInput) {
    studioDom.stepsInput.addEventListener("input", (e) => {
      studioState.steps = parseInt(e.target.value);
      if (studioDom.stepsVal) studioDom.stepsVal.innerText = studioState.steps;
      updateEstimates();
    });
  }

  const aspectButtons = getAspectButtons();
  if (aspectButtons) {
    aspectButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        aspectButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        studioState.ratio = btn.dataset.ratio;
        studioState.width = parseInt(btn.dataset.width);
        studioState.height = parseInt(btn.dataset.height);
        if (studioDom.resReadout)
          studioDom.resReadout.innerText = `${studioState.width} x ${studioState.height}`;
        updateEstimates();
      });
    });
  }

  if (studioDom.generateBtn)
    studioDom.generateBtn.addEventListener("click", handleGenerate);
  if (studioDom.upscaleBtn)
    studioDom.upscaleBtn.addEventListener("click", handleUpscale);
  if (studioDom.downloadBtn) {
    studioDom.downloadBtn.addEventListener("click", () =>
      downloadImage(studioDom.resultImg.src),
    );
  }

  // Gallery Clicks
  if (studioDom.gallery) {
    studioDom.gallery.addEventListener("click", (e) => {
      const item = e.target.closest(".gallery-item");
      if (item) selectGalleryItem(item.dataset.index);
    });
  }

  if (studioDom.sidebarGallery) {
    studioDom.sidebarGallery.addEventListener("click", (e) => {
      const item = e.target.closest(".gallery-item-sidebar");
      if (item) selectGalleryItem(item.dataset.index);
    });
  }

  // Lightbox
  if (studioDom.resultImg) {
    studioDom.resultImg.addEventListener("click", () => {
      studioDom.lightboxImg.src = studioDom.resultImg.src;
      studioDom.lightbox.classList.add("active");
    });
  }
  if (studioDom.lightboxClose) {
    studioDom.lightboxClose.addEventListener("click", () =>
      studioDom.lightbox.classList.remove("active"),
    );
  }
  if (studioDom.lightboxOverlay) {
    studioDom.lightboxOverlay.addEventListener("click", () =>
      studioDom.lightbox.classList.remove("active"),
    );
  }
}

// --- Logic ---

async function fetchHardwareStats() {
  console.log("Fetching hardware stats from backend...");
  if (studioDom.gpuName) studioDom.gpuName.innerText = "Checking hardware...";

  try {
    const resp = await fetch("/api/v1/system/config", {
      headers: { Authorization: `Bearer ${state.authToken}` },
    });
    if (resp.ok) {
      const config = await resp.json();
      console.log("Hardware config received:", config);
      studioState.hardware = config;

      if (studioDom.gpuName) {
        console.log("Setting GPU Name:", config.gpu_name);
        studioDom.gpuName.innerText =
          config.gpu_name || (config.use_gpu ? "NVIDIA GPU" : "CPU Mode");
      }

      if (studioDom.vramInfo) {
        studioDom.vramInfo.innerText = config.gpu_vram
          ? `${config.gpu_vram} GB VRAM`
          : "";
      }

      // Update resolution readout to match defaults
      if (studioDom.resReadout)
        studioDom.resReadout.innerText = `${studioState.width} x ${studioState.height}`;

      // Enforce resolution limits
      enforceHardwareConstraints(config);
    } else {
      console.warn("Hardware fetch returned status:", resp.status);
      if (studioDom.gpuName)
        studioDom.gpuName.innerText = "Local Hardware Mode";
    }
  } catch (e) {
    console.error("Hardware fetch failed", e);
    if (studioDom.gpuName) studioDom.gpuName.innerText = "Offline Mode";
  }
}

function enforceHardwareConstraints(config) {
  if (!config.use_gpu || config.gpu_vram < 6) {
    if (studioDom.aspectButtons) {
      studioDom.aspectButtons.forEach((btn) => {
        const w = parseInt(btn.dataset.width);
        const h = parseInt(btn.dataset.height);
        if (w > 1024 || h > 1024) {
          btn.disabled = true;
          btn.title = "Requires at least 6GB VRAM";
        }
      });
    }
  }
}

function updateEstimates() {
  let gpuScore = 30; // Default medium
  const gpuName = (studioState.hardware?.gpu_name || "").toLowerCase();

  if (
    gpuName.includes("4090") ||
    gpuName.includes("a100") ||
    gpuName.includes("h100")
  )
    gpuScore = 120;
  else if (gpuName.includes("4080") || gpuName.includes("3090")) gpuScore = 100;
  else if (gpuName.includes("4070") || gpuName.includes("3080")) gpuScore = 80;
  else if (gpuName.includes("4060") || gpuName.includes("3070")) gpuScore = 60;
  else if (gpuName.includes("3060")) gpuScore = 40;
  else if (!studioState.hardware?.use_gpu) gpuScore = 5; // CPU is slow

  const steps = studioState.steps;
  const resFactor = (studioState.width * studioState.height) / (1024 * 1024);

  const constant = 40; // Tuned constant
  const overhead = 1.0;

  let estPerImage = (steps * resFactor * constant) / gpuScore;

  if (studioState.model === "sdxl-turbo") estPerImage *= 0.8;
  if (studioState.model === "sdxl-lightning") estPerImage *= 1.2;

  const totalEst = estPerImage * studioState.batchSize + overhead;
  if (studioDom.timeEstimate)
    studioDom.timeEstimate.innerText = `~${totalEst.toFixed(1)}s`;
}

function handleFileSelect(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    studioState.inputImage = e.target.result.split(",")[1];
    if (studioDom.inputPreview) {
      studioDom.inputPreview.src = e.target.result;
      studioDom.inputPreview.style.display = "block";
    }
  };
  reader.readAsDataURL(file);
}

async function handleGenerate() {
  const prompt = studioDom.prompt?.value.trim();
  if (!prompt) {
    alert("Please enter a prompt");
    return;
  }

  setLoading(true, "Weaving pixels...");

  const results = [];
  try {
    for (let i = 0; i < studioState.batchSize; i++) {
      if (studioState.batchSize > 1 && studioDom.loadingText) {
        studioDom.loadingText.innerText = `Generating (${i + 1}/${studioState.batchSize})...`;
      }

      const payload = {
        prompt: prompt,
        negative_prompt: studioDom.negativePrompt?.value || "",
        width: studioState.width,
        height: studioState.height,
        steps: studioState.steps,
        model_type: studioState.model,
        image: studioState.inputImage,
        strength: 0.7,
      };

      let endpoint = "/api/v1/generate";
      let requestPayload = payload;

      if (studioState.mode === "image-to-video") {
        endpoint = "/api/v1/generate-video";
        requestPayload = {
          image: studioState.inputImage,
          motion_bucket_id: 127,
          noise_aug_strength: 0.02,
          fps: 7,
          num_frames: 25,
          num_inference_steps: 25,
          output_format: "mp4",
        };
      }

      const resp = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${state.authToken}`,
        },
        body: JSON.stringify(requestPayload),
      });

      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();

      if (studioState.mode === "image-to-video") {
        results.push({ type: "video", data: data.video_base64 });
      } else {
        results.push({ type: "image", data: data.image_base64 });
      }
    }

    displayBatchResults(results);
  } catch (e) {
    console.error(e);
    alert(`Generation failed: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

async function handleUpscale() {
  if (!studioState.lastResult) return;

  setLoading(true, "AI Upscaling (2x)...");
  try {
    const resp = await fetch("/api/v1/upscale", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.authToken}`,
      },
      body: JSON.stringify({ image: studioState.lastResult }),
    });

    if (!resp.ok) throw new Error("Upscale failed");
    const data = await resp.json();

    studioState.lastResult = data.image_base64;
    if (studioDom.resultImg)
      studioDom.resultImg.src = `data:image/jpeg;base64,${data.image_base64}`;
  } catch (e) {
    alert(e.message);
  } finally {
    setLoading(false);
  }
}

function displayBatchResults(results) {
  if (!results.length) return;

  results.forEach((res) => {
    studioState.gallery.unshift(res);
  });

  studioState.lastResult = results[0];
  updatePreview(results[0]);

  if (studioDom.placeholderState)
    studioDom.placeholderState.style.display = "none";
  if (studioDom.previewActions) studioDom.previewActions.style.display = "flex";

  renderGallery();
}

function updatePreview(item) {
  if (!item) return;

  if (item.type === "video") {
    if (studioDom.resultImg) studioDom.resultImg.style.display = "none";
    if (studioDom.resultVideo) {
      studioDom.resultVideo.src = `data:video/mp4;base64,${item.data}`;
      studioDom.resultVideo.style.display = "block";
    }
    // Highscaling/Upscaling doesn't apply to video in this UI yet
    if (studioDom.upscaleBtn) studioDom.upscaleBtn.style.display = "none";
  } else {
    const rawData = typeof item === "string" ? item : item.data;
    if (studioDom.resultVideo) studioDom.resultVideo.style.display = "none";
    if (studioDom.resultImg) {
      studioDom.resultImg.src = `data:image/jpeg;base64,${rawData}`;
      studioDom.resultImg.style.display = "block";
    }
    if (studioDom.upscaleBtn)
      studioDom.upscaleBtn.style.display = "inline-flex";
  }
}

function renderGallery() {
  if (studioDom.gallery) studioDom.gallery.innerHTML = "";
  if (studioDom.sidebarGallery) studioDom.sidebarGallery.innerHTML = "";

  studioState.gallery.forEach((item, idx) => {
    const isVideo = item.type === "video";
    const data = typeof item === "string" ? item : item.data;

    const galleryItem = document.createElement("div");
    galleryItem.className = `gallery-item ${idx === 0 ? "active" : ""}`;
    galleryItem.dataset.index = idx;

    if (isVideo) {
      galleryItem.innerHTML = `
            <div class="video-thumb-container">
                <span class="material-symbols-rounded video-icon">videocam</span>
                <div class="video-overlay">VIDEO</div>
            </div>
        `;
    } else {
      galleryItem.innerHTML = `<img src="data:image/jpeg;base64,${data}" alt="Generation ${idx}" />`;
    }

    if (studioDom.gallery) studioDom.gallery.appendChild(galleryItem);

    const sidebarItem = galleryItem.cloneNode(true);
    sidebarItem.className = "gallery-item-sidebar";
    if (studioDom.sidebarGallery)
      studioDom.sidebarGallery.appendChild(sidebarItem);
  });
}

function selectGalleryItem(idx) {
  const item = studioState.gallery[idx];
  studioState.lastResult = item;
  updatePreview(item);

  document
    .querySelectorAll(".gallery-item, .gallery-item-sidebar")
    .forEach((el) => {
      el.classList.toggle("active", el.dataset.index == idx);
    });
}

function setLoading(isLoading, text = "Generating...") {
  if (studioDom.loadingOverlay)
    studioDom.loadingOverlay.style.display = isLoading ? "flex" : "none";
  if (studioDom.loadingText) studioDom.loadingText.innerText = text;
  if (studioDom.generateBtn) studioDom.generateBtn.disabled = isLoading;
}

function downloadImage(src, name = "bipod-imagine.jpg") {
  const link = document.createElement("a");
  link.href = src;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// --- Init ---
document.addEventListener("DOMContentLoaded", init);
