import { state, dom } from "./state.js";
import { apiFetch } from "./api.js";
import { showToast, createWelcomeHero, closeSidebarOnMobile } from "./utils.js";
import { appendMessage } from "./message-renderer.js";
import {
  refreshComposerState,
  restoreComposerDraft,
  saveCurrentComposerDraft,
} from "./chat.js";

function ensureLoadingIndicator() {
  if (dom.loadingIndicator) return dom.loadingIndicator;
  if (!dom.chatWindow) return null;

  const indicator = document.createElement("div");
  indicator.id = "loading-indicator";
  indicator.className = "typing-indicator hidden";
  indicator.setAttribute("aria-live", "polite");
  indicator.innerHTML =
    '<span></span><span></span><span></span><div class="typing-label">Bipod is thinking</div>';
  dom.chatWindow.appendChild(indicator);
  return indicator;
}

function resetChatWindow() {
  if (!dom.chatWindow) return null;

  const indicator = ensureLoadingIndicator();
  if (indicator?.parentElement) {
    indicator.parentElement.removeChild(indicator);
  }

  dom.chatWindow.innerHTML = "";
  if (indicator) {
    dom.chatWindow.appendChild(indicator);
  }
  return indicator;
}

export function resolvePreferredConversationId() {
  const params = new URLSearchParams(window.location.search);
  const urlConvId = params.get("c");
  const savedConversationId = localStorage.getItem("bipod_current_conversation");

  if (urlConvId && state.conversations.some((c) => c.id === urlConvId)) {
    return urlConvId;
  }
  if (
    state.currentConversationId &&
    state.conversations.some((c) => c.id === state.currentConversationId)
  ) {
    return state.currentConversationId;
  }
  if (
    savedConversationId &&
    state.conversations.some((c) => c.id === savedConversationId)
  ) {
    return savedConversationId;
  }

  return state.conversations.length > 0 ? state.conversations[0].id : null;
}

export async function restoreConversationSelection() {
  const target = resolvePreferredConversationId();
  if (!target) return;

  const hasRenderedMessages = !!dom.chatWindow?.querySelector(".message");

  if (state.currentConversationId === target && hasRenderedMessages) {
    renderConversations();
    return;
  }

  await switchConversation(target);
}

export async function fetchConversations(options = {}) {
  const { restoreSelection = false } = options;
  try {
    const response = await apiFetch("/conversations");
    if (!response.ok) throw new Error("Failed to load conversations");
    state.conversations = await response.json();
    renderConversations();
    if (restoreSelection) {
      await restoreConversationSelection();
    }
  } catch (error) {
    console.error("Error loading conversations:", error);
  }
}

export async function createNewConversation(title = "New Conversation") {
  try {
    saveCurrentComposerDraft();
    const response = await apiFetch("/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    });
    const data = await response.json();
    state.currentConversationId = data.id;
    localStorage.setItem("bipod_current_conversation", data.id);
    await fetchConversations();
    await switchConversation(state.currentConversationId);
  } catch (e) {
    console.error("Failed to create conversation", e);
  }
}

export async function loadMessages(convId) {
  const requestId = ++state.activeMessagesRequestId;
  const indicator = resetChatWindow();
  if (indicator) indicator.classList.remove("hidden");

  try {
    const response = await apiFetch(`/conversations/${convId}/messages`);
    if (!response.ok) throw new Error("Failed to load messages");
    const messages = await response.json();

    if (
      requestId !== state.activeMessagesRequestId ||
      state.currentConversationId !== convId
    ) {
      return;
    }

    if (messages.length === 0) {
      const hero = createWelcomeHero();
      dom.chatWindow.insertBefore(hero, ensureLoadingIndicator());
    }
    messages.forEach((m) =>
      appendMessage(m.role === "assistant" ? "ai" : m.role, m.content, false, {
        messageId: m.id,
        attachments: m.attachments || [],
        createdAt: m.created_at,
      }),
    );
  } catch (e) {
    console.error("Failed to load messages", e);
    if (
      requestId === state.activeMessagesRequestId &&
      state.currentConversationId === convId
    ) {
      appendMessage(
        "system",
        "Failed to load this conversation. Please try switching to it again.",
        false,
      );
    }
  } finally {
    if (
      requestId === state.activeMessagesRequestId &&
      state.currentConversationId === convId
    ) {
      refreshComposerState();
      dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;
    }
  }
}

export async function renameConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  const conv = state.conversations.find((c) => c.id === id);
  const newName = prompt(
    "Rename conversation:",
    conv ? conv.title : "New Conversation",
  );
  if (!newName || newName === (conv && conv.title)) return;

  await apiFetch(`/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title: newName }),
  });
  await fetchConversations();
  if (state.currentConversationId === id) {
    dom.chatTitle.innerText = newName;
  }
}

export async function archiveConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  const password = prompt("Enter a password to archive this conversation:");
  if (!password) return;

  await apiFetch(`/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ is_archived: true, password: password }),
  });
  await fetchConversations();
}

export async function unarchiveConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  const password = prompt("Enter the archive password to unarchive:");
  if (!password) return;

  try {
    const verifyRes = await apiFetch(`/conversations/${id}/unlock`, {
      method: "POST",
      body: JSON.stringify({ password }),
    });
    if (verifyRes.ok) {
      await apiFetch(`/conversations/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ is_archived: false }),
      });
      await fetchConversations();
      showToast("Conversation unlocked");
    } else {
      showToast("Invalid password");
    }
  } catch (err) {
    showToast("Error unlocking");
  }
}

export async function deleteConversation(id, e) {
  if (e) {
    e.stopPropagation();
    e.preventDefault();
  }
  if (!confirm("Are you sure you want to delete this conversation?")) return;

  await apiFetch(`/conversations/${id}`, { method: "DELETE" });
  delete state.conversationDrafts[id];
  delete state.pendingConversationIds[id];
  if (state.currentConversationId === id) {
    state.currentConversationId = null;
    localStorage.removeItem("bipod_current_conversation");
    const url = new URL(window.location);
    url.searchParams.delete("c");
    history.replaceState(null, "", url);
    resetChatWindow();
    dom.chatWindow.insertBefore(createWelcomeHero(), ensureLoadingIndicator());
    dom.chatTitle.innerText = "New Chat";
  }
  await fetchConversations();
  if (!state.currentConversationId && state.conversations.length > 0) {
    await switchConversation(state.conversations[0].id);
  }
}

export function renderConversations() {
  if (!dom.historyContainer) return;
  dom.historyContainer.innerHTML = "";
  state.conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = `history-item ${c.id === state.currentConversationId ? "active" : ""}`;

    const titleSpan = document.createElement("span");
    titleSpan.className = "conv-title";
    titleSpan.textContent = `${c.is_archived ? "🔒 " : ""}${c.title}`;
    item.appendChild(titleSpan);

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "actions";

    const renameBtn = document.createElement("button");
    renameBtn.className = "action-btn";
    renameBtn.innerHTML = '<span class="material-symbols-rounded">edit</span>';
    renameBtn.onclick = (e) => renameConversation(c.id, e);
    actionsDiv.appendChild(renameBtn);

    const archiveBtn = document.createElement("button");
    archiveBtn.className = "action-btn";
    if (c.is_archived) {
      archiveBtn.innerHTML =
        '<span class="material-symbols-rounded">unarchive</span>';
      archiveBtn.onclick = (e) => unarchiveConversation(c.id, e);
    } else {
      archiveBtn.innerHTML =
        '<span class="material-symbols-rounded">archive</span>';
      archiveBtn.onclick = (e) => archiveConversation(c.id, e);
    }
    actionsDiv.appendChild(archiveBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "action-btn delete";
    deleteBtn.innerHTML =
      '<span class="material-symbols-rounded">delete</span>';
    deleteBtn.onclick = (e) => deleteConversation(c.id, e);
    actionsDiv.appendChild(deleteBtn);

    item.appendChild(actionsDiv);
    item.onclick = async () => {
      await switchConversation(c.id);
    };
    dom.historyContainer.appendChild(item);
  });
}

export async function switchConversation(id) {
  if (!id) return;
  if (state.currentConversationId !== id) {
    saveCurrentComposerDraft(state.currentConversationId);
  }
  const conv = state.conversations.find((c) => c.id === id);
  state.currentConversationId = id;
  localStorage.setItem("bipod_current_conversation", id);
  dom.chatTitle.innerText = conv ? conv.title : "Bipod";
  restoreComposerDraft(id, { focus: false });

  const url = new URL(window.location);
  url.searchParams.set("c", id);
  history.replaceState(null, "", url);

  renderConversations();
  await loadMessages(id);
  refreshComposerState();
  closeSidebarOnMobile();
}
