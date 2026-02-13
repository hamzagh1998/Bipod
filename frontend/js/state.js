export let state = {
  currentConversationId: null,
  conversations: [],
  authToken: localStorage.getItem("bipod_token"),
  currentUser: null,
  currentAttachments: [],
};

// --- Safe DOM Access Helper ---
const safeGet = (id) => document.getElementById(id);

export const dom = new Proxy(
  {},
  {
    get: (target, prop) => {
      // Basic elements that stay consistent across pages
      const commonElements = {
        sidebar: "sidebar",
        sidebarToggle: "sidebar-toggle",
        sidebarOverlay: "sidebar-overlay",
        logoutBtn: "logout-btn",
        currentUsernameSpan: "current-username",
        lightbox: "lightbox",
        lightboxImg: "lightbox-img",
        lightboxDownload: "lightbox-download",
        lightboxClose: "lightbox-close",
        lightboxOverlay: "lightbox-overlay",
      };

      if (commonElements[prop]) {
        return document.getElementById(commonElements[prop]);
      }

      // Chat specific elements
      const chatElements = {
        chatWindow: "chat-window",
        chatForm: "chat-form",
        userInput: "user-input",
        loadingIndicator: "loading-indicator",
        historyContainer: "history-container",
        newChatBtn: "new-chat-btn",
        chatTitle: "chat-title",
        welcomeHero: "welcome-hero",
        brainSettingsBtn: "brain-settings-btn",
        brainSettingsPanel: "brain-settings-panel",
        modelSelect: "model-select",
        modeSelect: "mode-select",
        imagineModelSelect: "imagine-model-select",
        attachBtn: "attach-btn",
        fileUpload: "file-upload",
        attachmentPreviewContainer: "attachment-preview-container",
        authOverlay: "auth-overlay",
        loginForm: "login-form",
        signupForm: "signup-form",
        toSignup: "to-signup",
        toLogin: "to-login",
      };

      if (chatElements[prop]) {
        return document.getElementById(chatElements[prop]);
      }

      return document.getElementById(prop);
    },
  },
);
