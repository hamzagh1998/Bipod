import { state, dom } from "./state.js";
import { setupMarkdown } from "./utils.js";
import { checkAuthStatus } from "./auth.js";
import { switchConversation } from "./conversations.js";
import { setupEventListeners } from "./ui.js";

async function init() {
  setupMarkdown();
  await checkAuthStatus();

  // Load Brain Settings
  const savedModel = localStorage.getItem("bipod_model");
  const savedMode = localStorage.getItem("bipod_mode");
  const savedImagine = localStorage.getItem("bipod_imagine_model");

  if (savedModel && dom.modelSelect) dom.modelSelect.value = savedModel;
  if (savedMode && dom.modeSelect) dom.modeSelect.value = savedMode;

  // Check hardware capabilities for Imagine models
  try {
    const configResp = await fetch("/api/v1/system/config", {
      headers: { Authorization: `Bearer ${state.authToken}` },
    });
    if (configResp.ok) {
      const config = await configResp.json();

      // Update hardware badge
      const badge = document.getElementById("hardware-info-badge");
      if (badge) {
        const gpuInfo = config.use_gpu
          ? `⚡ NVIDIA GPU (${config.gpu_vram}GB) detected`
          : "🧩 CPU Mode (No GPU detected)";
        badge.innerHTML = `<b>Local Hardware</b>${gpuInfo}<br>Tier: ${config.active_imagine_model === "stable-diffusion-xl" ? "Turbo/SDXL" : "Efficient"}`;
      }

      // Dynamic Model Selection & Validation
      if (dom.imagineModelSelect) {
        const xlOption = dom.imagineModelSelect.querySelector(
          'option[value="stable-diffusion-xl"]',
        );
        const sdOption = dom.imagineModelSelect.querySelector(
          'option[value="stable-diffusion"]',
        );

        if (!config.use_gpu) {
          if (xlOption) {
            xlOption.disabled = true;
            xlOption.innerText += " (GPU Required)";
          }
          if (sdOption) {
            sdOption.disabled = true;
            sdOption.innerText += " (GPU Required)";
          }
          dom.imagineModelSelect.value = "dalle-mini";
        } else if (config.gpu_vram < 5.5) {
          if (xlOption) {
            xlOption.disabled = true;
            xlOption.innerText += " (6GB VRAM Required)";
          }
          dom.imagineModelSelect.value = "stable-diffusion";
        } else {
          dom.imagineModelSelect.value = config.active_imagine_model;
        }

        if (savedImagine) {
          const opt = dom.imagineModelSelect.querySelector(
            `option[value="${savedImagine}"]`,
          );
          if (opt && !opt.disabled) {
            dom.imagineModelSelect.value = savedImagine;
          }
        }
      }

      // Auto-select the right Brain Model Tier
      if (config.active_brain_model && dom.modelSelect) {
        dom.modelSelect.value = config.active_brain_model;
      }
      if (savedModel && dom.modelSelect) dom.modelSelect.value = savedModel;
    }
  } catch (e) {
    console.error("Failed to fetch system config", e);
  }

  // Restore conversation from URL param
  const params = new URLSearchParams(window.location.search);
  const urlConvId = params.get("c");
  const target =
    urlConvId && state.conversations.find((c) => c.id === urlConvId)
      ? urlConvId
      : state.conversations.length > 0
        ? state.conversations[0].id
        : null;

  if (target) switchConversation(target);
  setupEventListeners();
}

// Boot the application
document.addEventListener("DOMContentLoaded", init);
