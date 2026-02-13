import { state, dom } from "./state.js";
import { apiFetch } from "./api.js";
import { showToast, createWelcomeHero } from "./utils.js";
import { appendMessage } from "./chat.js";
import { closeSidebarOnMobile } from "./ui.js";

export async function fetchConversations() {
  try {
    const response = await apiFetch("/conversations");
    if (!response.ok) throw new Error("Failed to load conversations");
    state.conversations = await response.json();
    renderConversations();
  } catch (error) {
    console.error("Error loading conversations:", error);
  }
}

export async function createNewConversation(title = "New Conversation") {
  try {
    const response = await apiFetch("/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    });
    const data = await response.json();
    state.currentConversationId = data.id;
    await fetchConversations();
    switchConversation(state.currentConversationId);
    closeSidebarOnMobile();
  } catch (e) {
    console.error("Failed to create conversation", e);
  }
}

export async function loadMessages(convId) {
  dom.chatWindow.innerHTML = "";
  dom.chatWindow.appendChild(dom.loadingIndicator);
  dom.loadingIndicator.classList.remove("hidden");

  try {
    const response = await apiFetch(`/conversations/${convId}/messages`);
    if (!response.ok) throw new Error("Failed to load messages");
    const messages = await response.json();
    if (messages.length === 0 && dom.welcomeHero) {
      const hero = createWelcomeHero();
      dom.chatWindow.insertBefore(hero, dom.loadingIndicator);
    }
    messages.forEach((m) =>
      appendMessage(m.role === "assistant" ? "ai" : m.role, m.content, false),
    );
  } catch (e) {
    console.error("Failed to load messages", e);
  } finally {
    dom.loadingIndicator.classList.add("hidden");
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
  if (state.currentConversationId === id) {
    state.currentConversationId = null;
    dom.chatWindow.innerHTML = "";
    dom.chatWindow.appendChild(dom.loadingIndicator);
    if (dom.welcomeHero) dom.welcomeHero.classList.remove("hidden");
    dom.chatTitle.innerText = "New Chat";
  }
  await fetchConversations();
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
    item.onclick = () => switchConversation(c.id);
    dom.historyContainer.appendChild(item);
  });
}

export function switchConversation(id) {
  const conv = state.conversations.find((c) => c.id === id);
  state.currentConversationId = id;
  dom.chatTitle.innerText = conv ? conv.title : "Bipod";

  const url = new URL(window.location);
  url.searchParams.set("c", id);
  history.replaceState(null, "", url);

  dom.loadingIndicator.classList.add("hidden");
  renderConversations();
  loadMessages(id);
  closeSidebarOnMobile();
}
