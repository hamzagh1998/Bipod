import { state, dom } from "./state.js";
import { apiFetch } from "./api.js";
import { showToast } from "./utils.js";
import { fetchConversations, renderConversations } from "./conversations.js";

export async function checkAuthStatus() {
  if (!state.authToken) {
    showAuth();
    return;
  }

  try {
    const res = await apiFetch("/auth/me");
    if (res.ok) {
      state.currentUser = await res.json();
      if (dom.currentUsernameSpan) {
        dom.currentUsernameSpan.innerText = state.currentUser.username;
      }
      hideAuth();
      await fetchConversations();
    } else {
      handleLogout();
    }
  } catch (err) {
    handleLogout();
  }
}

export function showAuth() {
  if (dom.authOverlay) {
    dom.authOverlay.classList.remove("hidden");
    dom.authOverlay.style.opacity = "1";
    dom.authOverlay.style.pointerEvents = "auto";
  }
}

export function hideAuth() {
  if (dom.authOverlay) {
    dom.authOverlay.style.opacity = "0";
    dom.authOverlay.style.pointerEvents = "none";
    setTimeout(() => dom.authOverlay.classList.add("hidden"), 400);
  }
}

export function handleLogout() {
  localStorage.removeItem("bipod_token");
  state.authToken = null;
  state.currentUser = null;
  state.currentConversationId = null;
  state.conversations = [];
  renderConversations();
  if (dom.chatWindow) {
    dom.chatWindow.innerHTML = "";
    dom.chatWindow.appendChild(dom.loadingIndicator);
  }
  showAuth();
}

export async function handleLogin(e) {
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
      state.authToken = data.access_token;
      localStorage.setItem("bipod_token", state.authToken);
      checkAuthStatus();
    } else {
      const err = await res.json();
      showToast(err.detail || "Login failed");
    }
  } catch (err) {
    showToast("Server unavailable");
  }
}

export async function handleSignup(e) {
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
      state.authToken = data.access_token;
      localStorage.setItem("bipod_token", state.authToken);
      checkAuthStatus();
    } else {
      const err = await res.json();
      showToast(err.detail || "Registration failed");
    }
  } catch (err) {
    showToast("Server unavailable");
  }
}
