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

  // Check hardware capabilities and populate models
  try {
    const configResp = await fetch("/api/v1/system/config", {
      headers: { Authorization: `Bearer ${state.authToken}` },
    });
    if (configResp.ok) {
      const config = await configResp.json();

      // 1. Update hardware badge
      const badge = document.getElementById("hardware-info-badge");
      if (badge) {
        const gpuInfo = config.use_gpu
          ? `⚡ ${config.gpu_name || "NVIDIA GPU"} (${config.gpu_vram}GB) detected`
          : "🧩 CPU Mode (No GPU detected)";

        let tierLabel = "Efficient";
        if (config.active_imagine_model === "flux-schnell")
          tierLabel = "Ultra (Flux)";
        else if (config.active_imagine_model === "sdxl-lightning")
          tierLabel = "High (Lightning)";

        badge.innerHTML = `<b>Local Hardware</b>${gpuInfo}<br>Tier: ${tierLabel}`;
      }

      // 2. Populate Brain Models
      if (dom.modelSelect && config.available_brain_models) {
        dom.modelSelect.innerHTML = "";
        config.available_brain_models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.textContent = `${m.name} — ${m.req}`;
          dom.modelSelect.appendChild(opt);
        });

        // Auto-select best brain
        dom.modelSelect.value = config.active_brain_model;
        if (savedModel) {
          const exists = Array.from(dom.modelSelect.options).some(
            (o) => o.value === savedModel,
          );
          if (exists) dom.modelSelect.value = savedModel;
        }
      }

      // 3. Populate Imagine Models
      if (dom.imagineModelSelect && config.available_imagine_models) {
        dom.imagineModelSelect.innerHTML = "";
        config.available_imagine_models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.textContent = `${m.name} (${m.req})`;
          // VRAM check for Flux / SDXL
          if (m.available === false) {
            opt.disabled = true;
            opt.textContent += " — HW Limit";
          }
          dom.imagineModelSelect.appendChild(opt);
        });

        // Auto-select best imagine
        dom.imagineModelSelect.value = config.active_imagine_model;
        if (savedImagine) {
          const opt = dom.imagineModelSelect.querySelector(
            `option[value="${savedImagine}"]`,
          );
          if (opt && !opt.disabled) dom.imagineModelSelect.value = savedImagine;
        }
      }
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
