// --- Global State ---
export let state = {
  currentConversationId: null,
  conversations: [],
  authToken: localStorage.getItem("bipod_token"),
  currentUser: null,
  currentImages: [],
};

// --- DOM References ---
export const dom = {
  chatWindow: document.getElementById("chat-window"),
  chatForm: document.getElementById("chat-form"),
  userInput: document.getElementById("user-input"),
  loadingIndicator: document.getElementById("loading-indicator"),
  historyContainer: document.getElementById("history-container"),
  newChatBtn: document.getElementById("new-chat-btn"),
  chatTitle: document.getElementById("chat-title"),
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  sidebarOverlay: document.getElementById("sidebar-overlay"),
  welcomeHero: document.getElementById("welcome-hero"),

  // Brain Settings
  brainSettingsBtn: document.getElementById("brain-settings-btn"),
  brainSettingsPanel: document.getElementById("brain-settings-panel"),
  modelSelect: document.getElementById("model-select"),
  modeSelect: document.getElementById("mode-select"),

  // Image Upload
  attachBtn: document.getElementById("attach-btn"),
  imageUpload: document.getElementById("image-upload"),
  imagePreviewContainer: document.getElementById("image-preview-container"),

  // Auth Overlay
  authOverlay: document.getElementById("auth-overlay"),
  loginForm: document.getElementById("login-form"),
  signupForm: document.getElementById("signup-form"),
  toSignup: document.getElementById("to-signup"),
  toLogin: document.getElementById("to-login"),
  logoutBtn: document.getElementById("logout-btn"),
  currentUsernameSpan: document.getElementById("current-username"),
};
