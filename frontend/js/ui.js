import { state, dom } from "./state.js";
import { createNewConversation } from "./conversations.js";
import { sendMessage } from "./chat.js";
import { renderImagePreviews } from "./images.js";
import { handleLogin, handleSignup, handleLogout } from "./auth.js";

export function setupEventListeners() {
  // Brain Settings Toggle
  dom.brainSettingsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    dom.brainSettingsPanel.classList.toggle("active");
    dom.brainSettingsBtn.classList.toggle("active");
  });

  // Close Settings on Click Outside
  document.addEventListener("click", (e) => {
    if (
      !dom.brainSettingsPanel.contains(e.target) &&
      !dom.brainSettingsBtn.contains(e.target)
    ) {
      dom.brainSettingsPanel.classList.remove("active");
      dom.brainSettingsBtn.classList.remove("active");
    }
  });

  // Save Settings on Change
  dom.modelSelect.addEventListener("change", () => {
    localStorage.setItem("bipod_model", dom.modelSelect.value);
  });
  dom.modeSelect.addEventListener("change", () => {
    localStorage.setItem("bipod_mode", dom.modeSelect.value);
  });

  // Sidebar Toggle
  dom.sidebarToggle.addEventListener("click", () => {
    if (window.innerWidth <= 768) {
      dom.sidebar.classList.toggle("mobile-open");
      dom.sidebarOverlay.classList.toggle("active");
    } else {
      dom.sidebar.classList.toggle("hidden");
    }
  });

  // Close sidebar when clicking overlay
  dom.sidebarOverlay.addEventListener("click", closeSidebarOnMobile);

  // New Chat
  dom.newChatBtn.addEventListener("click", () => createNewConversation());

  // Chat Form Submit
  dom.chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = dom.userInput.value.trim();
    if (text || state.currentImages.length > 0) {
      dom.userInput.value = "";
      dom.userInput.style.height = "auto";
      await sendMessage(text || " ");
    }
  });

  // Attach Image
  dom.attachBtn.addEventListener("click", () => dom.imageUpload.click());

  // Handle File Selection
  dom.imageUpload.addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const fullBase64 = e.target.result;
        const base64Content = fullBase64.split(",")[1];
        state.currentImages.push(base64Content);
        renderImagePreviews();
      };
      reader.readAsDataURL(file);
    });
  });

  // Handle resize
  window.addEventListener("resize", () => {
    if (window.innerWidth > 768) {
      dom.sidebar.classList.remove("mobile-open");
      dom.sidebarOverlay.classList.remove("active");
    }
  });

  // Textarea Auto-resize & Shortcuts
  dom.userInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
  });

  dom.userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (typeof dom.chatForm.requestSubmit === "function") {
        dom.chatForm.requestSubmit();
      } else {
        dom.chatForm.dispatchEvent(
          new Event("submit", { cancelable: true, bubbles: true }),
        );
      }
    }
  });

  // Global Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      createNewConversation();
    }
  });

  // Auth Forms
  dom.loginForm.onsubmit = handleLogin;
  dom.signupForm.onsubmit = handleSignup;
  dom.logoutBtn.onclick = handleLogout;

  // Auth Toggle
  dom.toSignup.addEventListener("click", (e) => {
    e.preventDefault();
    dom.loginForm.classList.add("hidden");
    dom.signupForm.classList.remove("hidden");
  });

  dom.toLogin.addEventListener("click", (e) => {
    e.preventDefault();
    dom.signupForm.classList.add("hidden");
    dom.loginForm.classList.remove("hidden");
  });
}

export function closeSidebarOnMobile() {
  if (window.innerWidth <= 768) {
    dom.sidebar.classList.remove("mobile-open");
    dom.sidebarOverlay.classList.remove("active");
  }
}
