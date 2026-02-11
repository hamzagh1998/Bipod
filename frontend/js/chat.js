import { state, dom } from "./state.js";
import { apiFetch } from "./api.js";
import { wrapCodeBlocks, copyToClipboard } from "./utils.js";
import { fetchConversations } from "./conversations.js";
import { renderAttachmentPreviews } from "./attachments.js";

export function appendMessage(role, text, shouldScroll = true) {
  const hero = document.getElementById("welcome-hero");
  if (hero) hero.remove();

  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";

  if (role === "ai") {
    // Check for generated image success message and append image markdown
    // Path inside docker is /app/data/generated/filename.jpg
    const imgMatch = text.match(/Saved to: .*?\/generated\/(.*?\.jpg)/);
    if (imgMatch) {
      const filename = imgMatch[1];
      // Append markdown image if not already present
      if (!text.includes(`(/generated/${filename})`)) {
        text += `\n\n![Generated Image](/generated/${filename})`;
      }
    }

    msgDiv.dataset.rawContent = text;
    contentDiv.innerHTML = marked.parse(text);
    wrapCodeBlocks(contentDiv);
    contentDiv.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
    });

    // Add Lightbox support to all images in the AI response
    contentDiv.querySelectorAll("img").forEach((img) => {
      img.addEventListener("click", () => {
        dom.lightboxImg.src = img.src;
        dom.lightbox.classList.add("active");
      });
    });

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "msg-actions";
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.innerHTML =
      '<span class="material-symbols-rounded">content_copy</span> Copy response';
    copyBtn.onclick = () => copyToClipboard(text);
    actionsDiv.appendChild(copyBtn);
    contentDiv.appendChild(actionsDiv);
  } else if (role === "system") {
    contentDiv.innerText = text;
  } else {
    msgDiv.dataset.rawContent = text;
    contentDiv.innerHTML = `<p>${text.replace(/\n/g, "<br>")}</p>`;

    // Show attachments in user message bubble
    if (state.currentAttachments.length > 0) {
      const attachmentsDiv = document.createElement("div");
      attachmentsDiv.className = "message-attachments";
      state.currentAttachments.forEach((att) => {
        if (att.type === "image") {
          const img = document.createElement("img");
          img.src = `data:image/jpeg;base64,${att.content}`;
          img.className = "msg-attachment-img";
          attachmentsDiv.appendChild(img);
        } else {
          const pdfIcon = document.createElement("div");
          pdfIcon.className = "msg-attachment-pdf";
          pdfIcon.innerHTML = `<span class="material-symbols-rounded">description</span><span>${att.name}</span>`;
          attachmentsDiv.appendChild(pdfIcon);
        }
      });
      contentDiv.appendChild(attachmentsDiv);
    }

    const userActions = document.createElement("div");
    userActions.className = "msg-actions user-msg-actions";

    const editBtn = document.createElement("button");
    editBtn.className = "msg-action-btn";
    editBtn.innerHTML =
      '<span class="material-symbols-rounded">edit</span> Edit';
    editBtn.onclick = () => {
      dom.userInput.value = text;
      dom.userInput.focus();
      dom.userInput.style.height = "auto";
      dom.userInput.style.height = dom.userInput.scrollHeight + "px";
    };
    userActions.appendChild(editBtn);

    const resendBtn = document.createElement("button");
    resendBtn.className = "msg-action-btn";
    resendBtn.innerHTML =
      '<span class="material-symbols-rounded">refresh</span> Resend';
    resendBtn.onclick = () => sendMessage(text);
    userActions.appendChild(resendBtn);

    contentDiv.appendChild(userActions);
  }

  msgDiv.appendChild(contentDiv);
  dom.chatWindow.insertBefore(msgDiv, dom.loadingIndicator);
  if (shouldScroll) dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;
}

export async function sendMessage(text) {
  if (!text && state.currentAttachments.length === 0) return;

  // Auto-create conversation if none selected
  if (!state.currentConversationId) {
    try {
      const response = await apiFetch("/conversations", {
        method: "POST",
        body: JSON.stringify({ title: "New Conversation" }),
      });
      const data = await response.json();
      state.currentConversationId = data.id;
      await fetchConversations();
    } catch (e) {
      console.error("Failed to auto-create conversation", e);
      appendMessage("system", "⚠ Failed to initialize conversation.");
      return;
    }
  }

  const sentForId = state.currentConversationId;
  appendMessage("user", text);
  dom.loadingIndicator.classList.remove("hidden");
  dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;

  try {
    const response = await apiFetch("/chat", {
      method: "POST",
      body: JSON.stringify({
        message: text,
        conversation_id: sentForId,
        model_id: dom.modelSelect.value,
        reasoning_mode: dom.modeSelect.value,
        imagine_model: dom.imagineModelSelect.value,
        attachments: state.currentAttachments,
      }),
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
