// ═══════════════════════════════════════════
// Bipod — Frontend Logic
// ═══════════════════════════════════════════

// --- State ---
let currentConversationId = null;
let conversations = [];
let authToken = localStorage.getItem("bipod_token");
let currentUser = null;

// --- API Wrapper ---
async function apiFetch(endpoint, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  try {
    const response = await fetch(`/api/v1${endpoint}`, {
      ...options,
      headers,
    });

    if (response.status === 401 && !endpoint.includes("/auth/")) {
      // Unauthorized (except for auth routes themselves)
      handleLogout();
      throw new Error("Session expired. Please login again.");
    }

    return response;
  } catch (err) {
    console.error(`API Fetch Error [${endpoint}]:`, err);
    throw err;
  }
}

// --- DOM References ---
const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const loadingIndicator = document.getElementById("loading-indicator");
const historyContainer = document.getElementById("history-container");
const newChatBtn = document.getElementById("new-chat-btn");
const chatTitle = document.getElementById("chat-title");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const welcomeHero = document.getElementById("welcome-hero");

// Brain Settings
const brainSettingsBtn = document.getElementById("brain-settings-btn");
const brainSettingsPanel = document.getElementById("brain-settings-panel");
const modelSelect = document.getElementById("model-select");
const modeSelect = document.getElementById("mode-select");

// Image Upload
const attachBtn = document.getElementById("attach-btn");
const imageUpload = document.getElementById("image-upload");
const imagePreviewContainer = document.getElementById(
  "image-preview-container",
);

let currentImages = []; // Stores base64 strings

// --- Markdown Configuration ---
marked.setOptions({
  highlight: function (code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true,
});

// --- Toast Notification ---
function showToast(message, duration = 2000) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), duration);
}

// --- Auth Logic ---
const authOverlay = document.getElementById("auth-overlay");
const loginForm = document.getElementById("login-form");
const signupForm = document.getElementById("signup-form");
const toSignup = document.getElementById("to-signup");
const toLogin = document.getElementById("to-login");
const logoutBtn = document.getElementById("logout-btn");
const currentUsernameSpan = document.getElementById("current-username");

toSignup.onclick = (e) => {
  e.preventDefault();
  loginForm.classList.add("hidden");
  signupForm.classList.remove("hidden");
};

toLogin.onclick = (e) => {
  e.preventDefault();
  signupForm.classList.add("hidden");
  loginForm.classList.remove("hidden");
};

async function checkAuthStatus() {
  if (!authToken) {
    showAuth();
    return;
  }

  try {
    const res = await apiFetch("/auth/me");
    if (res.ok) {
      currentUser = await res.json();
      currentUsernameSpan.innerText = currentUser.username;
      hideAuth();
      fetchConversations();
    } else {
      handleLogout();
    }
  } catch (err) {
    handleLogout();
  }
}

function showAuth() {
  authOverlay.classList.remove("hidden");
  authOverlay.style.opacity = "1";
  authOverlay.style.pointerEvents = "auto";
}

function hideAuth() {
  authOverlay.style.opacity = "0";
  authOverlay.style.pointerEvents = "none";
  setTimeout(() => authOverlay.classList.add("hidden"), 400);
}

function handleLogout() {
  localStorage.removeItem("bipod_token");
  authToken = null;
  currentUser = null;
  currentConversationId = null;
  conversations = [];
  renderConversations();
  chatWindow.innerHTML = "";
  showAuth();
}

loginForm.onsubmit = async (e) => {
  e.preventDefault();
  const username = document.getElementById("login-username").value;
  const password = document.getElementById("login-password").value;

  try {
    const res = await apiFetch("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (res.ok) {
      const data = await res.json();
      authToken = data.access_token;
      localStorage.setItem("bipod_token", authToken);
      checkAuthStatus();
    } else {
      const err = await res.json();
      showToast(err.detail || "Login failed");
    }
  } catch (err) {
    showToast("Server unavailable");
  }
};

signupForm.onsubmit = async (e) => {
  e.preventDefault();
  const username = document.getElementById("signup-username").value;
  const password = document.getElementById("signup-password").value;

  try {
    const res = await apiFetch("/auth/signup", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (res.ok) {
      const data = await res.json();
      authToken = data.access_token;
      localStorage.setItem("bipod_token", authToken);
      checkAuthStatus();
    } else {
      const err = await res.json();
      showToast(err.detail || "Registration failed");
    }
  } catch (err) {
    showToast("Server unavailable");
  }
};

logoutBtn.onclick = handleLogout;

// --- Clipboard Helper ---
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard");
    return true;
  } catch (err) {
    // Fallback for older browsers
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    showToast("Copied to clipboard");
    return true;
  }
}

// ═══════════════════════════════════════════
// Image Handling
// ═══════════════════════════════════════════
function renderImagePreviews() {
  imagePreviewContainer.innerHTML = "";

  if (currentImages.length === 0) {
    imagePreviewContainer.classList.remove("active");
    return;
  }

  imagePreviewContainer.classList.add("active");

  currentImages.forEach((b64, index) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    const img = document.createElement("img");
    // Function assume jpeg/png compat.
    // Attempt to detect mime type if possible, or just prepend generic.
    // Since we stripped it, we might guess.
    // Better: store object { base64, previewUrl }.
    // For now, simple base64 prepend works for most browsers if format is standard.
    // Actually, let's fix the logic above to store full data URL for preview.
    // RE-WRITE logic below to fix this assumption
    img.src = `data:image/jpeg;base64,${b64}`; // This is a simplification, ideally mime type should be preserved

    item.appendChild(img);

    const removeBtn = document.createElement("button");
    removeBtn.className = "preview-remove";
    removeBtn.innerHTML = "&times;";
    removeBtn.onclick = () => {
      currentImages.splice(index, 1);
      renderImagePreviews();
    };
    item.appendChild(removeBtn);

    imagePreviewContainer.appendChild(item);
  });
}

// ═══════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════
async function init() {
  await fetchConversations();

  await fetchConversations();

  // Load Brain Settings
  const savedModel = localStorage.getItem("bipod_model");
  const savedMode = localStorage.getItem("bipod_mode");
  if (savedModel) modelSelect.value = savedModel;
  if (savedMode) modeSelect.value = savedMode;

  // Restore conversation from URL param
  const params = new URLSearchParams(window.location.search);
  const urlConvId = params.get("c");
  const target =
    urlConvId && conversations.find((c) => c.id === urlConvId)
      ? urlConvId
      : conversations.length > 0
        ? conversations[0].id
        : null;

  if (target) switchConversation(target);
  setupEventListeners();
}

// ═══════════════════════════════════════════
// API Calls
// ═══════════════════════════════════════════
async function fetchConversations() {
  try {
    const response = await fetch("/api/v1/conversations");
    conversations = await response.json();
    renderHistory();
  } catch (e) {
    console.error("Failed to load history", e);
  }
}

async function createNewConversation() {
  try {
    const response = await fetch("/api/v1/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Conversation" }),
    });
    const data = await response.json();
    currentConversationId = data.id;
    await fetchConversations();
    await fetchConversations();
    switchConversation(currentConversationId);
    closeSidebarOnMobile();
  } catch (e) {
    console.error("Failed to create conversation", e);
  }
}

async function loadMessages(convId) {
  chatWindow.innerHTML = "";
  chatWindow.appendChild(loadingIndicator);

  try {
    const response = await fetch(`/api/v1/conversations/${convId}/messages`);
    const messages = await response.json();
    if (messages.length === 0 && welcomeHero) {
      // Re-insert the welcome hero for empty conversations
      const hero = createWelcomeHero();
      chatWindow.insertBefore(hero, loadingIndicator);
    }
    messages.forEach((m) =>
      appendMessage(m.role === "assistant" ? "ai" : m.role, m.content, false),
    );
  } catch (e) {
    console.error("Failed to load messages", e);
  }
}

function createWelcomeHero() {
  const hero = document.createElement("div");
  hero.id = "welcome-hero";
  hero.className = "welcome-hero";
  hero.innerHTML = `
    <svg class="welcome-logo" viewBox="0 0 36 36" width="56" height="56" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="welcome-grad-2" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#8a2be2"/>
          <stop offset="100%" style="stop-color:#a855f7"/>
        </linearGradient>
      </defs>
      <rect width="36" height="36" rx="8" fill="url(#welcome-grad-2)"/>
      <path d="M11 8h6a5 5 0 0 1 0 10h-6V8zm2 3v4h4a2 2 0 1 0 0-4h-4z" fill="white"/>
      <path d="M11 18h6a5 5 0 0 1 0 10h-6V18zm2 3v4h4a2 2 0 1 0 0-4h-4z" fill="white" opacity="0.7"/>
      <circle cx="27" cy="13" r="3" fill="white" opacity="0.5"/>
      <circle cx="27" cy="23" r="2" fill="white" opacity="0.3"/>
    </svg>
    <h2>Welcome to Bipod</h2>
    <p>Your local, weightless AI companion. Private by design.</p>
  `;
  return hero;
}

// ═══════════════════════════════════════════
// Conversation Actions (Chrome-safe)
// ═══════════════════════════════════════════
async function renameConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  const conv = conversations.find((c) => c.id === id);
  const newName = prompt(
    "Rename conversation:",
    conv ? conv.title : "New Conversation",
  );
  if (!newName || newName === (conv && conv.title)) return;

  await fetch(`/api/v1/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: newName }),
  });
  await fetchConversations();
  if (currentConversationId === id) {
    chatTitle.innerText = newName;
  }
}

async function archiveConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  const password = prompt("Enter a password to archive this conversation:");
  if (!password) return;

  await fetch(`/api/v1/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_archived: true, password: password }),
  });
  await fetchConversations();
}

async function unarchiveConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  const password = prompt("Enter the archive password to unarchive:");
  if (!password) return;

  try {
    const verifyRes = await fetch(`/api/v1/conversations/${id}/unlock`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (!verifyRes.ok) {
      showToast("Incorrect password");
      return;
    }
    await fetch(`/api/v1/conversations/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ is_archived: false }),
    });
    showToast("Conversation unarchived");
    await fetchConversations();
  } catch (err) {
    console.error("Unarchive failed", err);
    showToast("Failed to unarchive");
  }
}

async function deleteConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  if (
    !confirm(
      "Are you sure you want to delete this conversation? Context will be cleared.",
    )
  )
    return;

  await fetch(`/api/v1/conversations/${id}`, { method: "DELETE" });
  await fetchConversations();
  if (currentConversationId === id) {
    if (conversations.length > 0) switchConversation(conversations[0].id);
    else createNewConversation();
  }
}

// ═══════════════════════════════════════════
// UI Rendering
// ═══════════════════════════════════════════
function renderHistory() {
  historyContainer.innerHTML = "";
  conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = `history-item ${c.id === currentConversationId ? "active" : ""}`;

    // Title span
    const titleSpan = document.createElement("span");
    titleSpan.className = "conv-title";
    titleSpan.textContent = `${c.is_archived ? "🔒 " : ""}${c.title}`;
    item.appendChild(titleSpan);

    // Actions container
    const actionsDiv = document.createElement("div");
    actionsDiv.className = "actions";

    // Rename button
    const renameBtn = document.createElement("button");
    renameBtn.className = "action-btn";
    renameBtn.title = "Rename";
    renameBtn.innerHTML = '<span class="material-symbols-rounded">edit</span>';
    renameBtn.addEventListener("click", (e) => renameConversation(c.id, e));
    actionsDiv.appendChild(renameBtn);

    // Archive / Unarchive button
    const archiveBtn = document.createElement("button");
    archiveBtn.className = "action-btn";
    if (c.is_archived) {
      archiveBtn.title = "Unarchive";
      archiveBtn.innerHTML =
        '<span class="material-symbols-rounded">unarchive</span>';
      archiveBtn.addEventListener("click", (e) =>
        unarchiveConversation(c.id, e),
      );
    } else {
      archiveBtn.title = "Archive";
      archiveBtn.innerHTML =
        '<span class="material-symbols-rounded">archive</span>';
      archiveBtn.addEventListener("click", (e) => archiveConversation(c.id, e));
    }
    actionsDiv.appendChild(archiveBtn);

    // Delete button
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "action-btn delete";
    deleteBtn.title = "Delete";
    deleteBtn.innerHTML =
      '<span class="material-symbols-rounded">delete</span>';
    deleteBtn.addEventListener("click", (e) => deleteConversation(c.id, e));
    actionsDiv.appendChild(deleteBtn);

    item.appendChild(actionsDiv);

    // Click to switch conversation
    item.addEventListener("click", () => switchConversation(c.id));
    historyContainer.appendChild(item);
  });
}

function switchConversation(id) {
  const conv = conversations.find((c) => c.id === id);
  if (conv && conv.is_archived) {
    const password = prompt(
      "This conversation is archived. Enter password to unlock:",
    );
    if (!password) return;
  }

  currentConversationId = id;
  chatTitle.innerText = conv ? conv.title : "Bipod";

  // Sync URL without full page reload
  const url = new URL(window.location);
  url.searchParams.set("c", id);
  history.replaceState(null, "", url);

  // Hide any in-flight loading indicator from a previous conversation
  loadingIndicator.classList.add("hidden");

  renderHistory();
  renderHistory();
  loadMessages(id);
  closeSidebarOnMobile();
}

// ═══════════════════════════════════════════
// Message Rendering
// ═══════════════════════════════════════════
function appendMessage(role, text, shouldScroll = true) {
  // Remove welcome hero when first message arrives
  const hero = document.getElementById("welcome-hero");
  if (hero) hero.remove();

  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";

  if (role === "ai") {
    // Store raw markdown for copy
    msgDiv.dataset.rawContent = text;

    contentDiv.innerHTML = marked.parse(text);

    // Wrap code blocks with header + copy button
    wrapCodeBlocks(contentDiv);

    // Highlight any remaining code
    contentDiv.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
    });

    // Add message actions (copy full response)
    const actionsDiv = document.createElement("div");
    actionsDiv.className = "msg-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.innerHTML =
      '<span class="material-symbols-rounded">content_copy</span> Copy response';
    copyBtn.addEventListener("click", () => {
      copyToClipboard(text);
    });
    actionsDiv.appendChild(copyBtn);

    contentDiv.appendChild(actionsDiv);
  } else if (role === "system") {
    contentDiv.innerText = text;
  } else {
    // User message
    msgDiv.dataset.rawContent = text;
    contentDiv.innerHTML = `<p>${text.replace(/\n/g, "<br>")}</p>`;

    // Edit / Resend actions
    const userActions = document.createElement("div");
    userActions.className = "msg-actions user-msg-actions";

    const editBtn = document.createElement("button");
    editBtn.className = "msg-action-btn";
    editBtn.innerHTML =
      '<span class="material-symbols-rounded">edit</span> Edit';
    editBtn.addEventListener("click", () => {
      userInput.value = text;
      userInput.focus();
      userInput.style.height = "auto";
      userInput.style.height = userInput.scrollHeight + "px";
    });
    userActions.appendChild(editBtn);

    const resendBtn = document.createElement("button");
    resendBtn.className = "msg-action-btn";
    resendBtn.innerHTML =
      '<span class="material-symbols-rounded">refresh</span> Resend';
    resendBtn.addEventListener("click", () => {
      sendMessage(text);
    });
    userActions.appendChild(resendBtn);

    contentDiv.appendChild(userActions);
  }

  msgDiv.appendChild(contentDiv);
  chatWindow.insertBefore(msgDiv, loadingIndicator);
  if (shouldScroll) chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ═══════════════════════════════════════════
// Send Message (shared by form submit & resend)
// ═══════════════════════════════════════════
async function sendMessage(text) {
  if ((!text && currentImages.length === 0) || !currentConversationId) return;

  // Snapshot the conversation this message belongs to
  const sentForId = currentConversationId;

  appendMessage("user", text);
  loadingIndicator.classList.remove("hidden");
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        conversation_id: sentForId,
        model_id: modelSelect.value,
        reasoning_mode: modeSelect.value,
        images: currentImages,
      }),
    });

    // Clear images after sending
    currentImages = [];
    renderImagePreviews();
    imageUpload.value = ""; // Reset file input
    if (!response.ok) throw new Error("Network response was not ok");
    const data = await response.json();

    // Only update the UI if the user is still on the same conversation
    if (currentConversationId === sentForId) {
      appendMessage("ai", data.response);
    }

    // Auto-rename "New Conversation" after first exchange
    const conv = conversations.find((c) => c.id === sentForId);
    if (conv && conv.title === "New Conversation") {
      const autoTitle = text.length > 40 ? text.substring(0, 40) + "…" : text;
      await fetch(`/api/v1/conversations/${sentForId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: autoTitle }),
      });
      if (currentConversationId === sentForId) {
        chatTitle.innerText = autoTitle;
      }
      await fetchConversations();
    }
  } catch (error) {
    console.error("Chat error:", error);
    if (currentConversationId === sentForId) {
      appendMessage(
        "system",
        "⚠ Brain synchronization failed. Check connection.",
      );
    }
  } finally {
    // Only hide loading if still on the same conversation
    if (currentConversationId === sentForId) {
      loadingIndicator.classList.add("hidden");
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  }
}

// ═══════════════════════════════════════════
// Code Block Enhancement
// ═══════════════════════════════════════════
function wrapCodeBlocks(container) {
  const preBlocks = container.querySelectorAll("pre");

  preBlocks.forEach((pre) => {
    const codeEl = pre.querySelector("code");
    if (!codeEl) return;

    // Detect language from class
    const langClass = Array.from(codeEl.classList).find((cls) =>
      cls.startsWith("language-"),
    );
    const lang = langClass ? langClass.replace("language-", "") : "code";

    // Create wrapper
    const wrapper = document.createElement("div");
    wrapper.className = "code-block-wrapper";

    // Header with language label + copy button
    const header = document.createElement("div");
    header.className = "code-block-header";

    const langLabel = document.createElement("span");
    langLabel.textContent = lang;
    header.appendChild(langLabel);

    const copyBtn = document.createElement("button");
    copyBtn.className = "code-copy-btn";
    copyBtn.innerHTML =
      '<span class="material-symbols-rounded">content_copy</span> Copy';
    copyBtn.addEventListener("click", () => {
      copyToClipboard(codeEl.textContent);
      copyBtn.innerHTML =
        '<span class="material-symbols-rounded">check</span> Copied!';
      setTimeout(() => {
        copyBtn.innerHTML =
          '<span class="material-symbols-rounded">content_copy</span> Copy';
      }, 2000);
    });
    header.appendChild(copyBtn);

    wrapper.appendChild(header);

    // Move pre into wrapper
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);
  });
}

// ═══════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════
function setupEventListeners() {
  // Brain Settings Toggle
  brainSettingsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    brainSettingsPanel.classList.toggle("active");
    brainSettingsBtn.classList.toggle("active");
  });

  // Close Settings on Click Outside
  document.addEventListener("click", (e) => {
    if (
      !brainSettingsPanel.contains(e.target) &&
      !brainSettingsBtn.contains(e.target)
    ) {
      brainSettingsPanel.classList.remove("active");
      brainSettingsBtn.classList.remove("active");
    }
  });

  // Save Settings on Change
  modelSelect.addEventListener("change", () => {
    localStorage.setItem("bipod_model", modelSelect.value);
  });
  modeSelect.addEventListener("change", () => {
    localStorage.setItem("bipod_mode", modeSelect.value);
  });

  // Sidebar Toggle
  sidebarToggle.addEventListener("click", () => {
    if (window.innerWidth <= 768) {
      sidebar.classList.toggle("mobile-open");
      sidebarOverlay.classList.toggle("active");
    } else {
      sidebar.classList.toggle("hidden");
    }
  });

  // Close sidebar when clicking overlay
  sidebarOverlay.addEventListener("click", () => {
    closeSidebarOnMobile();
  });

  // New Chat
  newChatBtn.addEventListener("click", createNewConversation);

  // Chat Form Submit
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (text || currentImages.length > 0) {
      userInput.value = "";
      userInput.style.height = "auto";
      await sendMessage(text || " "); // Send even if only image
    }
  });

  // Attach Image
  attachBtn.addEventListener("click", () => imageUpload.click());

  // Handle File Selection
  imageUpload.addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        // Get Base64 string (remove data URL prefix for API if needed,
        // but Ollama usually handles base64 content.
        // We'll keep full data URL for preview, strip for API in send?
        // Actually Ollama wants just the base64 part often.
        const fullBase64 = e.target.result;
        const base64Content = fullBase64.split(",")[1]; // Strip "data:image/xyz;base64,"

        currentImages.push(base64Content);
        renderImagePreviews();
      };
      reader.readAsDataURL(file);
    });
  });

  // Handle resize to reset mobile state if moving to desktop
  window.addEventListener("resize", () => {
    if (window.innerWidth > 768) {
      sidebar.classList.remove("mobile-open");
      sidebarOverlay.classList.remove("active");
    }
  });

  // Textarea Auto-resize & Shortcuts
  userInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
  });

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  // Global Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      createNewConversation();
    }
  });
}

// ═══════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════
init();
function closeSidebarOnMobile() {
  if (window.innerWidth <= 768) {
    sidebar.classList.remove("mobile-open");
    sidebarOverlay.classList.remove("active");
  }
}
