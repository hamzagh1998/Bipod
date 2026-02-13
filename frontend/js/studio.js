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
  console.log("🚀 Bipod Studio Init starting...");

  // Ensure libraries are loaded before calling setup
  if (typeof marked !== "undefined" && typeof hljs !== "undefined") {
    setupMarkdown();
  } else {
    console.warn(
      "Markdown/Highlight.js not found. Deferring UI utility setup.",
    );
  }

  try {
    await checkAuthStatus();
  } catch (err) {
    console.error("Auth check failed:", err);
  }

  if (!state.authToken) {
    window.location.href = "/";
    return;
  }

  // Set username
  if (state.currentUser && studioDom.currentUsernameSpan) {
    studioDom.currentUsernameSpan.innerText = state.currentUser.username;
  }

  setupEventListeners();

  // Don't await this so UI can finish init faster
  fetchHardwareStats().finally(() => {
    updateEstimates();
  });

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
        if (
          studioState.model === "sdxl-lightning" ||
          studioState.model === "flux-schnell"
        ) {
          studioDom.stepsInput.value = 4;
        } else if (studioState.model === "sdxl-turbo") {
          studioDom.stepsInput.value = 1;
        } else {
          studioDom.stepsInput.value = 30;
        }
        studioDom.stepsInput.dispatchEvent(new Event("input"));
      }
      updateModelCapabilities();
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

        // Show upload group for img2img
        if (studioDom.imgUploadGroup) {
          studioDom.imgUploadGroup.style.display =
            studioState.mode === "image-to-image" ? "block" : "none";
        }
        updateEstimates();
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

      // Populate models
      if (config.available_imagine_models) {
        populateModels(config.available_imagine_models);
      }
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
    const aspectButtons = getAspectButtons();
    if (aspectButtons) {
      aspectButtons.forEach((btn) => {
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

function populateModels(models) {
  if (!studioDom.modelSelect) return;

  const currentVal = studioDom.modelSelect.value;
  studioDom.modelSelect.innerHTML = "";

  models.forEach((model) => {
    // Skip Flux if not available (VRAM constraint)
    if (model.id === "flux-schnell" && model.available === false) {
      return;
    }

    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name;
    studioDom.modelSelect.appendChild(option);
  });

  // Try to restore previous value if it still exists
  if ([...studioDom.modelSelect.options].some((o) => o.value === currentVal)) {
    studioDom.modelSelect.value = currentVal;
  } else if (studioDom.modelSelect.options.length > 0) {
    studioState.model = studioDom.modelSelect.value;
  }

  updateModelCapabilities();
}

function updateModelCapabilities() {
  const selectedModelId = studioDom.modelSelect?.value;
  if (!selectedModelId || !studioState.hardware?.available_imagine_models)
    return;

  const model = studioState.hardware.available_imagine_models.find(
    (m) => m.id === selectedModelId,
  );
  if (!model) return;

  // Toggle Negative Prompt
  if (studioDom.negativePrompt) {
    const group = studioDom.negativePrompt.closest(".control-group");
    if (group) {
      group.style.opacity =
        model.supports_negative_prompt === false ? "0.5" : "1";
      studioDom.negativePrompt.disabled =
        model.supports_negative_prompt === false;
      if (model.supports_negative_prompt === false) {
        studioDom.negativePrompt.title = "Not supported by this model";
      } else {
        studioDom.negativePrompt.title = "";
      }
    }
  }

  // Toggle Img2Img / Mode Buttons
  const modeButtons = getModeButtons();
  modeButtons.forEach((btn) => {
    if (btn.dataset.mode === "image-to-image") {
      btn.disabled = model.supports_img2img === false;
      btn.title =
        model.supports_img2img === false ? "Not supported by this model" : "";

      // If we were in img2img and it's now disabled, switch to text-to-image
      if (studioState.mode === "image-to-image" && btn.disabled) {
        const t2iBtn = [...modeButtons].find(
          (b) => b.dataset.mode === "text-to-image",
        );
        if (t2iBtn) t2iBtn.click();
      }
    }

    // Video mode - handle specially as it's often separate
    if (btn.dataset.mode === "image-to-video") {
      // Only enable if we have a video model (like svd-xt) - though the user removed it for now
      const hasVideoModel = studioState.hardware.available_imagine_models.some(
        (m) => m.id === "svd-xt",
      );
      btn.disabled = !hasVideoModel;
      btn.style.display = hasVideoModel ? "block" : "none";
    }
  });
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
  else if (gpuName.includes("4050") || gpuName.includes("3060")) gpuScore = 40;
  else if (gpuName.includes("3050")) gpuScore = 30;
  else if (!studioState.hardware?.use_gpu) gpuScore = 5; // CPU is slow

  const steps = studioState.steps;
  const resFactor = (studioState.width * studioState.height) / (1024 * 1024);

  const constant = 40; // Tuned constant
  const overhead = 1.0;

  let estPerImage = (steps * resFactor * constant) / gpuScore;

  if (studioState.model === "sdxl-turbo") estPerImage *= 0.8;
  if (studioState.model === "sdxl-lightning") estPerImage *= 1.2;
  if (studioState.model === "flux-schnell") estPerImage *= 4.5; // Flux is significantly heavier

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

      results.push({ type: "image", data: data.image_base64 });
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
