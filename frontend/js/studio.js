import { state } from "./state.js";
import { setupMarkdown } from "./utils.js";
import { checkAuthStatus, handleLogout } from "./auth.js";

// --- Studio State ---
const studioState = {
  model: "sdxl-lightning",
  mode: "text-to-image",
  ratio: "1:1",
  templateWidth: 1024,
  templateHeight: 1024,
  resolutionTarget: 1024,
  width: 1024,
  height: 1024,
  batchSize: 1,
  steps: 4,
  inputImage: null,
  upscaleSourceImage: null,
  hardware: null,
  projects: [],
  currentProjectId: localStorage.getItem("bipod_studio_project"),
  currentPreviewItem: null,
  lastResult: null,
  lastResultMime: "image/jpeg",
  lastResultExtension: "jpg",
  upscaleFactor: 2,
  upscaler: "realesrgan",
  outputFormat: "jpeg",
  gallery: [],
};

// Initial DOM pointers
const _domMap = {
  sidebarToggle: "sidebar-toggle",
  sidebar: "sidebar",
  sidebarOverlay: "sidebar-overlay",
  logoutBtn: "logout-btn",
  currentUsernameSpan: "current-username",
  newProjectBtn: "studio-new-project-btn",
  projectList: "studio-project-list",
  gpuName: "gpu-name",
  vramInfo: "vram-info",
  modelSelect: "studio-model",
  imgUploadGroup: "studio-img-upload",
  dropZone: "studio-drop-zone",
  fileInput: "studio-file-input",
  upscaleDropZone: "studio-upscale-drop-zone",
  upscaleFileInput: "studio-upscale-file-input",
  inputPreview: "studio-input-preview",
  prompt: "studio-prompt",
  negativePrompt: "studio-negative-prompt",
  improvePromptBtn: "studio-improve-prompt-btn",
  improveNegativeBtn: "studio-improve-negative-btn",
  resReadout: "studio-res-readout",
  resolutionHint: "studio-resolution-hint",
  resolutionInput: "studio-resolution",
  resolutionVal: "studio-resolution-val",
  upscalerSelect: "studio-upscaler",
  outputFormatSelect: "studio-output-format",
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
  upscaleControls: "studio-upscale-controls",
  downloadBtn: "studio-download-btn",
  deleteImageBtn: "studio-delete-image-btn",
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
const getUpscaleFactorButtons = () =>
  document.querySelectorAll(".upscale-factor-btn");

let previewActionsRevealTimer = null;

function isTouchPreviewMode() {
  return window.matchMedia("(hover: none)").matches;
}

function parseSizeString(size) {
  const match = /^(\d+)x(\d+)$/.exec(size || "");
  if (!match) return null;
  return {
    width: parseInt(match[1], 10),
    height: parseInt(match[2], 10),
  };
}

function recommendUpscaleFactor(meta) {
  const requested = parseSizeString(meta?.requested_size);
  const actual = parseSizeString(meta?.actual_size);
  if (!requested || !actual) return 2;

  const requestedLongest = Math.max(requested.width, requested.height);
  const actualLongest = Math.max(actual.width, actual.height);
  if (requestedLongest <= actualLongest) return 2;

  const ratio = requestedLongest / actualLongest;
  if (ratio >= 6) return 8;
  if (ratio >= 3) return 4;
  return 2;
}

function resolutionEdge(size) {
  const parsed = parseSizeString(size);
  if (!parsed) return null;
  return Math.max(parsed.width, parsed.height);
}

function getSelectedImagineModel() {
  const models = studioState.hardware?.available_imagine_models || [];
  return models.find((model) => model.id === studioState.model) || null;
}

function canUseEightXUpscale() {
  return studioState.upscaler === "realesrgan";
}

function getResolutionInputValues() {
  if (!studioDom.resolutionInput) return [];
  return Array.from(studioDom.resolutionInput.options || [])
    .filter((option) => !option.disabled && !option.hidden)
    .map((option) => parseInt(option.value, 10))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
}

function chooseClosestResolutionOption(targetEdge) {
  const values = getResolutionInputValues();
  if (!values.length) return targetEdge || 1024;
  if (!targetEdge)
    return values[
      Math.min(
        values.length - 1,
        values.indexOf(1024) >= 0 ? values.indexOf(1024) : 0,
      )
    ];

  const atOrBelow = values.filter((value) => value <= targetEdge);
  if (atOrBelow.length) {
    return atOrBelow[atOrBelow.length - 1];
  }
  return values[0];
}

function getModelDefaultResolutionTarget(model) {
  const recommendedEdge = resolutionEdge(model?.recommended_resolution);
  return chooseClosestResolutionOption(recommendedEdge || 1024);
}

function setResolutionTarget(value) {
  if (!Number.isFinite(value)) return;
  studioState.resolutionTarget = value;
  if (studioDom.resolutionInput) {
    studioDom.resolutionInput.value = String(value);
  }
  if (studioDom.resolutionVal) {
    studioDom.resolutionVal.innerText = String(value);
  }
}

function syncResolutionInputConstraints(model) {
  if (!studioDom.resolutionInput) return;

  const modelMaxEdge = resolutionEdge(model?.max_resolution);
  const options = Array.from(studioDom.resolutionInput.options || []);

  options.forEach((option) => {
    const value = parseInt(option.value, 10);
    const allowed = !modelMaxEdge || value <= modelMaxEdge;
    option.disabled = !allowed;
    option.hidden = !allowed;
  });

  if (modelMaxEdge && (studioState.resolutionTarget || 0) > modelMaxEdge) {
    setResolutionTarget(chooseClosestResolutionOption(modelMaxEdge));
  }
}

function syncResolutionGuidance(model) {
  if (!studioDom.resolutionHint) return;

  const recommendedEdge = resolutionEdge(model?.recommended_resolution);
  const maxEdge = resolutionEdge(model?.max_resolution);
  const vramUsage = model?.vram_usage || model?.req || "Varies by hardware";

  if (!recommendedEdge && !maxEdge) {
    studioDom.resolutionHint.innerText = `Generate at a safe native size, then use 2x/4x/8x upscale for larger exports. Expected VRAM: ${vramUsage}.`;
    return;
  }

  const recommendedLabel = recommendedEdge
    ? `${recommendedEdge}px`
    : "the recommended size";
  const maxLabel = maxEdge ? `${maxEdge}px` : recommendedLabel;
  studioDom.resolutionHint.innerText = `Default native target ${recommendedLabel}; max native ${maxLabel}; expected VRAM ${vramUsage}. Use upscale for larger exports.`;
}

function syncModelDefaults(model) {
  if (!model) return;
  setResolutionTarget(getModelDefaultResolutionTarget(model));
  applyResolutionSelection();
}

function syncPreviewActionAvailability() {
  if (studioDom.deleteImageBtn) {
    studioDom.deleteImageBtn.disabled = !studioState.currentPreviewItem?.id;
  }
}

function clearPreview() {
  studioState.currentPreviewItem = null;
  studioState.lastResult = null;
  studioState.lastResultMime = "image/jpeg";
  studioState.lastResultExtension = "jpg";
  if (studioDom.resultImg) {
    studioDom.resultImg.src = "";
    studioDom.resultImg.style.display = "none";
  }
  if (studioDom.resultVideo) {
    studioDom.resultVideo.src = "";
    studioDom.resultVideo.style.display = "none";
  }
  if (studioDom.placeholderState) {
    studioDom.placeholderState.style.display = "flex";
  }
  if (studioDom.previewActions) {
    studioDom.previewActions.style.display = "none";
  }
  syncPreviewActionAvailability();
}

function mapSavedImageToGalleryItem(
  savedImage,
  base64Data = null,
  meta = null,
) {
  return {
    type: "image",
    id: savedImage.id,
    projectId: savedImage.project_id,
    filename: savedImage.filename,
    url: savedImage.url,
    data: base64Data,
    meta,
    mime: savedImage.mime_type || "image/jpeg",
    extension: savedImage.file_extension || "jpg",
    width: savedImage.width || null,
    height: savedImage.height || null,
    createdAt: savedImage.created_at || null,
  };
}

async function fetchStudioProjects() {
  const resp = await fetch("/api/v1/studio/projects", {
    headers: { Authorization: `Bearer ${state.authToken}` },
  });
  if (!resp.ok) throw new Error(await resp.text());

  studioState.projects = await resp.json();
  renderProjects();

  const preferredId = studioState.currentProjectId;
  const hasPreferred =
    preferredId &&
    studioState.projects.some((project) => project.id === preferredId);
  if (hasPreferred) {
    await selectStudioProject(preferredId, { persist: false });
    return;
  }
  if (studioState.projects.length) {
    await selectStudioProject(studioState.projects[0].id);
    return;
  }
  clearPreview();
  if (studioDom.sidebarGallery) {
    studioDom.sidebarGallery.innerHTML = "";
  }
  if (studioDom.gallery) {
    studioDom.gallery.innerHTML = "";
  }
}

function renderProjects() {
  if (!studioDom.projectList) return;
  studioDom.projectList.innerHTML = "";

  studioState.projects.forEach((project) => {
    const item = document.createElement("div");
    item.className = `studio-project-item ${project.id === studioState.currentProjectId ? "active" : ""}`;
    item.dataset.projectId = project.id;
    item.innerHTML = `
      <span class="material-symbols-rounded">folder</span>
      <div class="studio-project-meta">
        <div class="studio-project-title">${project.title}</div>
        <div class="studio-project-count">${project.image_count || 0} images</div>
      </div>
      <button class="icon-btn studio-project-delete" data-project-id="${project.id}" title="Delete Project">
        <span class="material-symbols-rounded">delete</span>
      </button>
    `;
    studioDom.projectList.appendChild(item);
  });
}

async function createStudioProject() {
  const title = window.prompt("Project name");
  if (!title || !title.trim()) return;

  const resp = await fetch("/api/v1/studio/projects", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${state.authToken}`,
    },
    body: JSON.stringify({ title: title.trim() }),
  });
  if (!resp.ok) {
    alert(await resp.text());
    return;
  }

  const project = await resp.json();
  studioState.projects.unshift(project);
  renderProjects();
  await selectStudioProject(project.id);
}

async function selectStudioProject(projectId, { persist = true } = {}) {
  studioState.currentProjectId = projectId;
  if (persist) {
    localStorage.setItem("bipod_studio_project", projectId);
  }
  renderProjects();
  await fetchProjectImages(projectId);
}

async function fetchProjectImages(projectId) {
  const resp = await fetch(`/api/v1/studio/projects/${projectId}/images`, {
    headers: { Authorization: `Bearer ${state.authToken}` },
  });
  if (!resp.ok) {
    alert(await resp.text());
    return;
  }

  const images = await resp.json();
  studioState.gallery = images.map((image) =>
    mapSavedImageToGalleryItem(image),
  );
  renderGallery();

  if (studioState.gallery.length) {
    selectGalleryItem(0);
  } else {
    clearPreview();
  }
}

async function deleteStudioProject(projectId) {
  const project = studioState.projects.find((entry) => entry.id === projectId);
  const label = project?.title || "this project";
  if (!window.confirm(`Delete ${label} and all of its images?`)) {
    return;
  }

  const resp = await fetch(`/api/v1/studio/projects/${projectId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${state.authToken}` },
  });
  if (!resp.ok) {
    alert(await resp.text());
    return;
  }

  studioState.projects = studioState.projects.filter(
    (entry) => entry.id !== projectId,
  );
  if (studioState.currentProjectId === projectId) {
    studioState.currentProjectId = null;
    localStorage.removeItem("bipod_studio_project");
  }
  renderProjects();

  if (studioState.projects.length) {
    await selectStudioProject(studioState.projects[0].id);
  } else {
    studioState.gallery = [];
    renderGallery();
    clearPreview();
  }
}

async function ensureCurrentPreviewBase64() {
  if (studioState.lastResult) return studioState.lastResult;
  const item = studioState.currentPreviewItem;
  if (!item?.url) return null;

  const resp = await fetch(item.url, {
    headers: { Authorization: `Bearer ${state.authToken}` },
  });
  if (!resp.ok) throw new Error("Failed to load image data for upscale.");
  const blob = await resp.blob();

  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result || "";
      const [, base64Payload = ""] = String(result).split(",", 2);
      studioState.lastResult = base64Payload;
      resolve(base64Payload);
    };
    reader.onerror = () =>
      reject(reader.error || new Error("Failed to read image data."));
    reader.readAsDataURL(blob);
  });
}

async function deleteCurrentPreviewImage() {
  const item = studioState.currentPreviewItem;
  if (!item?.id || !studioState.currentProjectId) return;
  if (!window.confirm("Delete this image from the current project?")) {
    return;
  }

  const resp = await fetch(
    `/api/v1/studio/projects/${studioState.currentProjectId}/images/${item.id}`,
    {
      method: "DELETE",
      headers: { Authorization: `Bearer ${state.authToken}` },
    },
  );
  if (!resp.ok) {
    alert(await resp.text());
    return;
  }

  studioState.gallery = studioState.gallery.filter(
    (entry) => entry.id !== item.id,
  );
  renderGallery();
  const project = studioState.projects.find(
    (entry) => entry.id === studioState.currentProjectId,
  );
  if (project) {
    project.image_count = Math.max(0, (project.image_count || 1) - 1);
    if (project.cover_image_url === item.url) {
      project.cover_image_url = studioState.gallery[0]?.url || null;
    }
  }
  renderProjects();

  if (studioState.gallery.length) {
    selectGalleryItem(0);
  } else {
    clearPreview();
  }
}

function setPromptImproveButtonState(
  button,
  isLoading,
  defaultLabel = "AI Improve",
) {
  if (!button) return;
  button.disabled = isLoading;
  button.innerText = isLoading ? "Improving..." : defaultLabel;
}

async function improvePromptField(targetField) {
  const promptValue = studioDom.prompt?.value.trim() || "";
  const negativeValue = studioDom.negativePrompt?.value.trim() || "";

  if (targetField === "prompt" && !promptValue) {
    alert("Enter a prompt first.");
    return;
  }
  if (targetField === "negative" && !promptValue && !negativeValue) {
    alert("Enter a prompt or a negative prompt first.");
    return;
  }

  const targetButton =
    targetField === "prompt"
      ? studioDom.improvePromptBtn
      : studioDom.improveNegativeBtn;
  const otherButton =
    targetField === "prompt"
      ? studioDom.improveNegativeBtn
      : studioDom.improvePromptBtn;

  setPromptImproveButtonState(targetButton, true);
  if (otherButton) {
    otherButton.disabled = true;
  }

  try {
    const resp = await fetch("/api/v1/studio/prompt-improve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.authToken}`,
      },
      body: JSON.stringify({
        prompt: promptValue,
        negative_prompt: negativeValue,
        model_type: studioState.model,
      }),
    });

    if (!resp.ok) {
      throw new Error(await resp.text());
    }

    const data = await resp.json();
    if (targetField === "prompt" && studioDom.prompt) {
      studioDom.prompt.value = data.prompt || promptValue;
    }
    if (targetField === "negative" && studioDom.negativePrompt) {
      studioDom.negativePrompt.value = data.negative_prompt || negativeValue;
    }
  } catch (error) {
    console.error("Prompt improvement failed", error);
    alert(`Prompt improvement failed: ${error.message}`);
  } finally {
    setPromptImproveButtonState(targetButton, false);
    if (otherButton) {
      otherButton.disabled = false;
    }
  }
}

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
  syncUpscaleControls();
  applyResolutionSelection();
  syncPreviewActionAvailability();

  try {
    await fetchStudioProjects();
  } catch (err) {
    console.error("Failed to load studio projects", err);
  }

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

  if (studioDom.newProjectBtn) {
    studioDom.newProjectBtn.addEventListener("click", createStudioProject);
  }

  // Controls
  if (studioDom.modelSelect) {
    // Sync initial model
    studioState.model = studioDom.modelSelect.value || "sdxl-lightning";
    studioDom.modelSelect.addEventListener("change", (e) => {
      studioState.model = e.target.value;
      if (studioDom.stepsInput) {
        if (studioState.model === "flux-schnell") {
          studioDom.stepsInput.value = 4;
          studioDom.stepsInput.dispatchEvent(new Event("input"));
        }
      }
      updateModelCapabilities();
      updateEstimates();
    });
  }

  if (studioDom.upscalerSelect) {
    studioState.upscaler = studioDom.upscalerSelect.value || "realesrgan";
    studioDom.upscalerSelect.addEventListener("change", (e) => {
      studioState.upscaler = e.target.value || "realesrgan";
      syncUpscaleControls();
    });
  }

  if (studioDom.outputFormatSelect) {
    studioState.outputFormat = studioDom.outputFormatSelect.value || "jpeg";
    studioDom.outputFormatSelect.addEventListener("change", (e) => {
      studioState.outputFormat = e.target.value || "jpeg";
    });
  }

  if (studioDom.improvePromptBtn) {
    studioDom.improvePromptBtn.addEventListener("click", () =>
      improvePromptField("prompt"),
    );
  }

  if (studioDom.improveNegativeBtn) {
    studioDom.improveNegativeBtn.addEventListener("click", () =>
      improvePromptField("negative"),
    );
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

  if (studioDom.upscaleDropZone) {
    studioDom.upscaleDropZone.addEventListener("click", () =>
      studioDom.upscaleFileInput.click(),
    );
  }

  if (studioDom.upscaleFileInput) {
    studioDom.upscaleFileInput.addEventListener("change", (e) =>
      handleStandaloneUpscaleFileSelect(e.target.files[0]),
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

  if (studioDom.resolutionInput) {
    const onResolutionChange = (e) => {
      setResolutionTarget(parseInt(e.target.value, 10));
      applyResolutionSelection();
      updateEstimates();
    };
    studioDom.resolutionInput.addEventListener("change", onResolutionChange);
    studioDom.resolutionInput.addEventListener("input", onResolutionChange);
  }

  const aspectButtons = getAspectButtons();
  if (aspectButtons) {
    aspectButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        aspectButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        studioState.ratio = btn.dataset.ratio;
        studioState.templateWidth = parseInt(btn.dataset.width);
        studioState.templateHeight = parseInt(btn.dataset.height);
        applyResolutionSelection();
        updateEstimates();
      });
    });
  }

  if (studioDom.generateBtn)
    studioDom.generateBtn.addEventListener("click", handleGenerate);
  const upscaleFactorButtons = getUpscaleFactorButtons();
  if (upscaleFactorButtons.length) {
    upscaleFactorButtons.forEach((btn) => {
      btn.addEventListener("click", async () => {
        let factor = parseInt(btn.dataset.upscaleFactor, 10) || 2;
        if (factor === 8 && !canUseEightXUpscale()) {
          alert(
            "8x upscale is only available with Real-ESRGAN. Switch the Upscale Engine to Real-ESRGAN to use 8x.",
          );
          return;
        }
        studioState.upscaleFactor = factor;
        syncUpscaleControls();
        if (studioState.lastResult) {
          await handleUpscale();
        }
      });
    });
  }
  if (studioDom.downloadBtn) {
    studioDom.downloadBtn.addEventListener("click", () =>
      downloadImage(studioDom.resultImg.src),
    );
  }
  if (studioDom.deleteImageBtn) {
    studioDom.deleteImageBtn.addEventListener(
      "click",
      deleteCurrentPreviewImage,
    );
  }
  if (studioDom.lightboxDownload) {
    studioDom.lightboxDownload.addEventListener("click", () =>
      downloadImage(studioDom.lightboxImg?.src || studioDom.resultImg?.src),
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

  if (studioDom.projectList) {
    studioDom.projectList.addEventListener("click", async (e) => {
      const deleteBtn = e.target.closest(".studio-project-delete");
      if (deleteBtn) {
        e.stopPropagation();
        await deleteStudioProject(deleteBtn.dataset.projectId);
        return;
      }

      const item = e.target.closest(".studio-project-item");
      if (item) {
        await selectStudioProject(item.dataset.projectId);
      }
    });
  }

  // Lightbox
  if (studioDom.mainPreview) {
    studioDom.mainPreview.addEventListener("click", (e) => {
      if (!isTouchPreviewMode() || e.target !== studioDom.mainPreview) return;
      const hasImage = studioDom.resultImg?.style.display !== "none";
      const hasVideo = studioDom.resultVideo?.style.display !== "none";
      if (
        (hasImage || hasVideo) &&
        !studioDom.mainPreview.classList.contains("controls-visible")
      ) {
        revealPreviewActions();
      }
    });
  }
  if (studioDom.resultImg) {
    studioDom.resultImg.addEventListener("click", () => {
      if (
        isTouchPreviewMode() &&
        !studioDom.mainPreview?.classList.contains("controls-visible")
      ) {
        revealPreviewActions();
        return;
      }
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

function revealPreviewActions(durationMs = 2200) {
  if (!studioDom.mainPreview) return;
  studioDom.mainPreview.classList.add("controls-visible");
  if (previewActionsRevealTimer) {
    clearTimeout(previewActionsRevealTimer);
  }
  previewActionsRevealTimer = window.setTimeout(() => {
    studioDom.mainPreview?.classList.remove("controls-visible");
    previewActionsRevealTimer = null;
  }, durationMs);
}

function syncUpscaleControls() {
  const upscalerLabel =
    studioState.upscaler === "realesrgan" ? "Real-ESRGAN" : "Swin2SR";
  const eightXAllowed = canUseEightXUpscale();
  if (!eightXAllowed && studioState.upscaleFactor === 8) {
    studioState.upscaleFactor = 4;
  }
  getUpscaleFactorButtons().forEach((btn) => {
    const factor = parseInt(btn.dataset.upscaleFactor, 10) || 2;
    if (factor === 8) {
      btn.disabled = !eightXAllowed;
      btn.classList.toggle("upscale-factor-btn--restricted", !eightXAllowed);
    } else {
      btn.disabled = false;
      btn.classList.remove("upscale-factor-btn--restricted");
    }
    btn.classList.toggle("active", factor === studioState.upscaleFactor);
    btn.setAttribute(
      "aria-pressed",
      factor === studioState.upscaleFactor ? "true" : "false",
    );
    if (factor === 8 && !eightXAllowed) {
      btn.title = "8x upscale is available only with Real-ESRGAN.";
      btn.classList.remove("upscale-factor-btn--capped");
    } else {
      btn.title = `${upscalerLabel} ${factor}x`;
      btn.classList.remove("upscale-factor-btn--capped");
    }
  });
}

function alignDimension(dim, alignment = 8) {
  return Math.max(alignment, Math.round(dim / alignment) * alignment);
}

function applyResolutionSelection() {
  const scale = (studioState.resolutionTarget || 1024) / 1024;
  let width = Math.round(studioState.templateWidth * scale);
  let height = Math.round(studioState.templateHeight * scale);

  const maxDim = 4096;
  const largest = Math.max(width, height);
  let clampEdge = maxDim;
  const modelMaxEdge = resolutionEdge(
    getSelectedImagineModel()?.max_resolution,
  );
  if (modelMaxEdge) {
    clampEdge = Math.min(clampEdge, modelMaxEdge);
  }

  if (largest > clampEdge) {
    const clampScale = clampEdge / largest;
    width = Math.round(width * clampScale);
    height = Math.round(height * clampScale);
  }

  studioState.width = alignDimension(width, 8);
  studioState.height = alignDimension(height, 8);

  if (studioDom.resReadout) {
    studioDom.resReadout.innerText = `${studioState.width} x ${studioState.height}`;
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

      if (studioDom.resolutionInput) {
        studioDom.resolutionInput.value = String(studioState.resolutionTarget);
      }
      if (studioDom.resolutionVal) {
        studioDom.resolutionVal.innerText = String(
          studioState.resolutionTarget,
        );
      }
      applyResolutionSelection();

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
  if (studioDom.resolutionInput && (!config.use_gpu || config.gpu_vram < 6)) {
    studioDom.resolutionInput.title =
      "Higher requested resolutions may fall back to a smaller size if CUDA runs out of memory.";
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

  syncResolutionInputConstraints(model);
  syncModelDefaults(model);
  syncResolutionGuidance(model);
  syncUpscaleControls();

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
  if (studioState.model === "juggernaut-xl") estPerImage *= 1.4;
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

function handleStandaloneUpscaleFileSelect(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const fullDataUrl = e.target.result;
    const rawBase64 = fullDataUrl.split(",")[1];
    studioState.upscaleSourceImage = rawBase64;
    studioState.lastResult = rawBase64;
    studioState.lastResultMime = file.type || "image/png";
    studioState.lastResultExtension = file.name?.includes(".")
      ? file.name.split(".").pop().toLowerCase()
      : "png";
    updatePreview({
      type: "image",
      data: rawBase64,
      meta: null,
      mime: studioState.lastResultMime,
      extension: studioState.lastResultExtension,
    });
    if (studioDom.placeholderState) {
      studioDom.placeholderState.style.display = "none";
    }
    if (studioDom.previewActions) {
      studioDom.previewActions.style.display = "flex";
    }
    revealPreviewActions(2600);
  };
  reader.readAsDataURL(file);
}

async function handleGenerate() {
  const prompt = studioDom.prompt?.value.trim();
  if (!prompt) {
    alert("Please enter a prompt");
    return;
  }
  if (!studioState.currentProjectId) {
    alert("Create or select a project first.");
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
        output_format: studioState.outputFormat,
        project_id: studioState.currentProjectId,
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
      const savedImage = data.saved_image;

      results.push({
        type: "image",
        data: data.image_base64,
        meta: data,
        ...(savedImage
          ? mapSavedImageToGalleryItem(savedImage, data.image_base64, data)
          : {
              mime: data.mime_type || "image/jpeg",
              extension: data.file_extension || "jpg",
            }),
      });
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
  if (!studioState.currentProjectId) {
    alert("Create or select a project first.");
    return;
  }
  const sourceImage = await ensureCurrentPreviewBase64();
  if (!sourceImage) return;

  const scale = studioState.upscaleFactor || 2;
  setLoading(true, `AI Upscaling (${scale}x)...`);
  try {
    const resp = await fetch("/api/v1/upscale", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.authToken}`,
      },
      body: JSON.stringify({
        image: sourceImage,
        scale,
        upscaler: studioState.upscaler,
        output_format: studioState.outputFormat,
        project_id: studioState.currentProjectId,
      }),
    });

    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    studioState.lastResult = data.image_base64;
    studioState.lastResultMime = data.mime_type || "image/jpeg";
    studioState.lastResultExtension = data.file_extension || "jpg";
    studioState.upscaleFactor = data.scale || scale;
    syncUpscaleControls();
    if (data.saved_image) {
      const item = mapSavedImageToGalleryItem(
        data.saved_image,
        data.image_base64,
        data,
      );
      studioState.gallery.unshift(item);
      const project = studioState.projects.find(
        (entry) => entry.id === studioState.currentProjectId,
      );
      if (project) {
        project.image_count = (project.image_count || 0) + 1;
        project.cover_image_url = data.saved_image.url;
      }
      renderProjects();
      renderGallery();
      selectGalleryItem(0);
    } else if (studioDom.resultImg) {
      studioDom.resultImg.src = `data:${studioState.lastResultMime};base64,${data.image_base64}`;
    }
    if (data.was_capped) {
      alert(
        "8x upscale is only available with Real-ESRGAN. The request was reduced to 4x.",
      );
    }
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
  const project = studioState.projects.find(
    (entry) => entry.id === studioState.currentProjectId,
  );
  if (project) {
    project.image_count = (project.image_count || 0) + results.length;
    project.cover_image_url =
      studioState.gallery[0]?.url || project.cover_image_url || null;
  }
  renderProjects();

  studioState.lastResult = results[0].type === "video" ? null : results[0].data;
  studioState.lastResultMime =
    results[0].type === "video" ? null : results[0].mime || "image/jpeg";
  studioState.lastResultExtension =
    results[0].type === "video" ? null : results[0].extension || "jpg";
  updatePreview(results[0]);

  if (studioDom.placeholderState)
    studioDom.placeholderState.style.display = "none";
  if (studioDom.previewActions) studioDom.previewActions.style.display = "flex";

  renderGallery();
}

function updatePreview(item) {
  if (!item) return;
  studioState.currentPreviewItem = item;

  if (item.type === "video") {
    studioDom.mainPreview?.classList.remove("controls-visible");
    if (studioDom.resultImg) studioDom.resultImg.style.display = "none";
    if (studioDom.resultVideo) {
      studioDom.resultVideo.src = `data:video/mp4;base64,${item.data}`;
      studioDom.resultVideo.style.display = "block";
    }
    // Highscaling/Upscaling doesn't apply to video in this UI yet
    if (studioDom.upscaleControls)
      studioDom.upscaleControls.style.display = "none";
  } else {
    const rawData = typeof item === "string" ? item : item.data;
    const meta = typeof item === "string" ? null : item.meta;
    const mime =
      (typeof item === "object" && item?.mime) ||
      studioState.lastResultMime ||
      "image/jpeg";
    const extension =
      (typeof item === "object" && item?.extension) ||
      studioState.lastResultExtension ||
      "jpg";
    const src =
      (typeof item === "object" && item?.url) ||
      (rawData ? `data:${mime};base64,${rawData}` : "");
    studioState.lastResult = rawData;
    studioState.lastResultMime = mime;
    studioState.lastResultExtension = extension;
    if (studioDom.resultVideo) studioDom.resultVideo.style.display = "none";
    if (studioDom.resultImg) {
      studioDom.resultImg.src = src;
      studioDom.resultImg.style.display = "block";
    }
    if (meta) {
      studioState.upscaleFactor = recommendUpscaleFactor(meta);
      syncUpscaleControls();
    }
    if (studioDom.upscaleControls)
      studioDom.upscaleControls.style.display = "flex";
  }
  if (studioDom.placeholderState) {
    studioDom.placeholderState.style.display = "none";
  }
  if (studioDom.previewActions) {
    studioDom.previewActions.style.display = "flex";
  }
  syncPreviewActionAvailability();
}

function renderGallery() {
  if (studioDom.gallery) studioDom.gallery.innerHTML = "";
  if (studioDom.sidebarGallery) studioDom.sidebarGallery.innerHTML = "";

  studioState.gallery.forEach((item, idx) => {
    const isVideo = item.type === "video";
    const data = typeof item === "string" ? item : item.data;
    const isActive =
      item?.id && studioState.currentPreviewItem?.id
        ? item.id === studioState.currentPreviewItem.id
        : idx === 0 && !studioState.currentPreviewItem;

    const galleryItem = document.createElement("div");
    galleryItem.className = `gallery-item ${isActive ? "active" : ""}`;
    galleryItem.dataset.index = idx;

    if (isVideo) {
      galleryItem.innerHTML = `
            <div class="video-thumb-container">
                <span class="material-symbols-rounded video-icon">videocam</span>
                <div class="video-overlay">VIDEO</div>
            </div>
        `;
    } else {
      const mime = item.mime || "image/jpeg";
      const src = item.url || `data:${mime};base64,${data}`;
      galleryItem.innerHTML = `<img src="${src}" alt="Generation ${idx}" />`;
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
  if (!item) {
    clearPreview();
    return;
  }
  studioState.lastResult =
    item && item.type === "video"
      ? null
      : typeof item === "string"
        ? item
        : item.data;
  studioState.lastResultMime =
    item && item.type === "video" ? null : item?.mime || "image/jpeg";
  studioState.lastResultExtension =
    item && item.type === "video" ? null : item?.extension || "jpg";
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
  getUpscaleFactorButtons().forEach((btn) => {
    btn.disabled = isLoading;
  });
  if (studioDom.downloadBtn) studioDom.downloadBtn.disabled = isLoading;
}

function downloadImage(src, name = null) {
  const link = document.createElement("a");
  link.href = src;
  link.download =
    name || `bipod-imagine.${studioState.lastResultExtension || "jpg"}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// --- Init ---
document.addEventListener("DOMContentLoaded", init);
