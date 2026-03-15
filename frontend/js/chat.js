import { state, dom } from "./state.js";
import { apiFetch } from "./api.js";
import { fetchConversations } from "./conversations.js";
import { renderAttachmentPreviews } from "./attachments.js";
import { appendMessage } from "./message-renderer.js";
import { showToast } from "./utils.js";

function cloneAttachments(attachments = []) {
  return attachments.map((attachment) => ({ ...attachment }));
}

function getDraftKey(conversationId = state.currentConversationId) {
  return conversationId || "__new__";
}

function getDraft(conversationId = state.currentConversationId) {
  const key = getDraftKey(conversationId);
  if (!state.conversationDrafts[key]) {
    state.conversationDrafts[key] = { text: "", attachments: [] };
  }
  return state.conversationDrafts[key];
}

function setDraft(conversationId, text = "", attachments = []) {
  state.conversationDrafts[getDraftKey(conversationId)] = {
    text,
    attachments: cloneAttachments(attachments),
  };
}

export function isConversationPending(
  conversationId = state.currentConversationId,
) {
  return !!(conversationId && state.pendingConversationIds[conversationId]);
}

function setConversationPending(conversationId, isPending) {
  if (!conversationId) return;
  if (isPending) {
    state.pendingConversationIds[conversationId] = true;
  } else {
    delete state.pendingConversationIds[conversationId];
  }
}

export function refreshComposerState() {
  const isPending = isConversationPending();
  if (dom.loadingIndicator) {
    dom.loadingIndicator.classList.toggle("hidden", !isPending);
  }
  if (dom.userInput) dom.userInput.disabled = isPending;
  if (dom.sendBtn) dom.sendBtn.disabled = isPending;
  if (dom.attachBtn) dom.attachBtn.disabled = isPending;
  if (dom.chatForm) dom.chatForm.classList.toggle("is-pending", isPending);
}

export function saveCurrentComposerDraft(
  conversationId = state.currentConversationId,
) {
  setDraft(
    conversationId,
    dom.userInput ? dom.userInput.value : "",
    state.currentAttachments,
  );
}

export function setComposerDraft(text = "", attachments = [], options = {}) {
  const { focus = true } = options;
  state.currentAttachments = cloneAttachments(attachments);
  renderAttachmentPreviews();

  if (dom.userInput) {
    dom.userInput.value = text || "";
    dom.userInput.style.height = "auto";
    dom.userInput.style.height = `${dom.userInput.scrollHeight}px`;
    if (focus) {
      dom.userInput.focus();
      dom.userInput.setSelectionRange(
        dom.userInput.value.length,
        dom.userInput.value.length,
      );
    }
  }

  setDraft(state.currentConversationId, text || "", attachments);
}

export function restoreComposerDraft(
  conversationId = state.currentConversationId,
  options = {},
) {
  const draft = getDraft(conversationId);
  setComposerDraft(draft.text, draft.attachments, options);
}

export async function resendMessage(text, attachments = []) {
  return sendMessage(text, { attachments });
}

export async function sendMessage(text, options = {}) {
  console.log("Chat: sendMessage triggered with text:", text);
  if (isConversationPending()) {
    showToast("Wait for the current response to finish.");
    return;
  }

  const outgoingAttachments = cloneAttachments(
    options.attachments ?? state.currentAttachments,
  );
  if (!text && outgoingAttachments.length === 0) return;

  // Check if marked is available
  if (typeof marked === "undefined") {
    console.error("CRITICAL: marked.js library not loaded!");
    appendMessage(
      "system",
      "⚠ System error: Markdown renderer missing. Refreshing may help.",
    );
    return;
  }

  // Auto-create conversation if none selected
  if (!state.currentConversationId) {
    console.log("Chat: No conversation ID, attempting auto-create...");
    try {
      const pendingDraft = getDraft(null);
      const response = await apiFetch("/conversations", {
        method: "POST",
        body: JSON.stringify({ title: "New Conversation" }),
      });
      const data = await response.json();
      state.currentConversationId = data.id;
      localStorage.setItem("bipod_current_conversation", data.id);
      setDraft(data.id, pendingDraft.text, pendingDraft.attachments);
      delete state.conversationDrafts.__new__;
      const url = new URL(window.location);
      url.searchParams.set("c", data.id);
      history.replaceState(null, "", url);
      if (dom.chatTitle) dom.chatTitle.innerText = "New Conversation";
      await fetchConversations();
    } catch (e) {
      console.error("Chat: Failed to auto-create conversation:", e);
      appendMessage("system", "⚠ Failed to initialize conversation.");
      return;
    }
  }

  const sentForId = state.currentConversationId;
  console.log("Conversation ID:", sentForId); // DEBUG
  appendMessage("user", text, true, { attachments: outgoingAttachments });

  setDraft(sentForId, "", []);
  if (state.currentConversationId === sentForId) {
    state.currentAttachments = [];
    renderAttachmentPreviews();
    if (dom.fileUpload) dom.fileUpload.value = "";
  }

  setConversationPending(sentForId, true);
  refreshComposerState();
  if (dom.chatWindow) dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;

  try {
    const payload = {
      message: text,
      conversation_id: sentForId,
      model_id: dom.modelSelect ? dom.modelSelect.value : null,
      reasoning_mode: dom.modeSelect ? dom.modeSelect.value : "normal",
      imagine_model: dom.imagineModelSelect
        ? dom.imagineModelSelect.value
        : "sdxl-lightning",
      attachments: outgoingAttachments,
    };
    console.log("Payload:", payload); // DEBUG

    const response = await apiFetch("/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error("Network response was not ok");
    const data = await response.json();

    if (state.currentConversationId === sentForId) {
      appendMessage("ai", data.response);
    }

    const conv = state.conversations.find((c) => c.id === sentForId);
    if (conv && conv.title === "New Conversation") {
      const autoTitle = text.length > 40 ? text.substring(0, 40) + "…" : text;
      await apiFetch(`/conversations/${sentForId}`, {
        method: "PATCH",
        body: JSON.stringify({ title: autoTitle }),
      });
      if (state.currentConversationId === sentForId) {
        dom.chatTitle.innerText = autoTitle;
      }
      await fetchConversations();
    }
  } catch (error) {
    console.error("Chat error:", error);
    if (state.currentConversationId === sentForId) {
      appendMessage(
        "system",
        "⚠ Brain synchronization failed. Check connection.",
      );
    }
  } finally {
    setConversationPending(sentForId, false);
    refreshComposerState();
    if (state.currentConversationId === sentForId && dom.chatWindow) {
      dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;
    }
  }
}

export function downloadImage(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || "bipod_image.jpg";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
