import { state, dom } from "./state.js";
import { setupMarkdown } from "./utils.js";
import { checkAuthStatus } from "./auth.js";
import { restoreConversationSelection } from "./conversations.js";
import { setupEventListeners } from "./ui.js";

// Global error handling for easier debugging
window.onerror = function (msg, url, lineNo, columnNo, error) {
  console.error("GLOBAL ERROR:", msg, "at", url, ":", lineNo, ":", columnNo);
  return false;
};

async function init() {
  console.log("Bipod initializing...");
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
          if (m.available === false) {
            opt.disabled = true;
            opt.textContent += " — Not recommended here";
          }
          dom.modelSelect.appendChild(opt);
        });

        // Auto-select best brain
        dom.modelSelect.value = config.active_brain_model;
        if (savedModel) {
          const savedOption = Array.from(dom.modelSelect.options).find(
            (o) => o.value === savedModel,
          );
          if (savedOption && !savedOption.disabled) {
            dom.modelSelect.value = savedModel;
          } else {
            localStorage.removeItem("bipod_model");
          }
        }
      }

      // 3. Populate Imagine Models
      if (dom.imagineModelSelect && config.available_imagine_models) {
        dom.imagineModelSelect.innerHTML = "";
        config.available_imagine_models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          const requirementLabel =
            m.req ||
            (typeof m.requires_vram_gb === "number"
              ? `${m.requires_vram_gb}+ GB VRAM`
              : null) ||
            m.vram_usage ||
            null;
          opt.textContent = requirementLabel
            ? `${m.name} (${requirementLabel})`
            : m.name;
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

  await restoreConversationSelection();
  setupEventListeners();
}

// Boot the application
document.addEventListener("DOMContentLoaded", init);
