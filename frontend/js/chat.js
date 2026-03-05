import { state, dom } from "./state.js";
import { apiFetch } from "./api.js";
import { fetchConversations } from "./conversations.js";
import { renderAttachmentPreviews } from "./attachments.js";
import { appendMessage } from "./message-renderer.js";

export async function sendMessage(text) {
  console.log("Chat: sendMessage triggered with text:", text);
  if (!text && state.currentAttachments.length === 0) return;

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
      const response = await apiFetch("/conversations", {
        method: "POST",
        body: JSON.stringify({ title: "New Conversation" }),
      });
      const data = await response.json();
      state.currentConversationId = data.id;
      await fetchConversations();
    } catch (e) {
      console.error("Chat: Failed to auto-create conversation:", e);
      appendMessage("system", "⚠ Failed to initialize conversation.");
      return;
    }
  }

  const sentForId = state.currentConversationId;
  console.log("Conversation ID:", sentForId); // DEBUG
  appendMessage("user", text);

  if (dom.loadingIndicator) dom.loadingIndicator.classList.remove("hidden");
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
      attachments: state.currentAttachments,
    };
    console.log("Payload:", payload); // DEBUG

    const response = await apiFetch("/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    state.currentAttachments = [];
    renderAttachmentPreviews();
    dom.fileUpload.value = "";
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
    if (state.currentConversationId === sentForId) {
      dom.loadingIndicator.classList.add("hidden");
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
