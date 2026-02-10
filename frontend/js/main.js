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
  if (savedModel) dom.modelSelect.value = savedModel;
  if (savedMode) dom.modeSelect.value = savedMode;

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
