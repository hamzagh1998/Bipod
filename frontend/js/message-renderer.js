import { state, dom } from "./state.js";
import {
  wrapCodeBlocks,
  escapeHtml,
  copyToClipboard,
  showToast,
} from "./utils.js";

let userMessageActionHandler = null;

export function setUserMessageActionHandler(handler) {
  userMessageActionHandler = handler;
}

function renderAttachments(container, attachments = []) {
  if (!attachments || attachments.length === 0) return;

  const attachmentsDiv = document.createElement("div");
  attachmentsDiv.className = "message-attachments";

  attachments.forEach((att) => {
    if (att.type === "image") {
      const img = document.createElement("img");
      img.src = `data:image/jpeg;base64,${att.content}`;
      img.className = "msg-attachment-img";
      attachmentsDiv.appendChild(img);
    } else {
      const pdfIcon = document.createElement("div");
      pdfIcon.className = "msg-attachment-pdf";
      pdfIcon.innerHTML = `<span class="material-symbols-rounded">description</span><span>${escapeHtml(att.name || "PDF")}</span>`;
      attachmentsDiv.appendChild(pdfIcon);
    }
  });

  container.appendChild(attachmentsDiv);
}

function isSafeUrl(url, { allowDataImage = false } = {}) {
  if (!url) return false;
  const normalized = String(url).trim().toLowerCase();
  if (normalized.startsWith("javascript:")) return false;
  if (normalized.startsWith("data:")) {
    return allowDataImage && normalized.startsWith("data:image/");
  }
  return (
    normalized.startsWith("http://") ||
    normalized.startsWith("https://") ||
    normalized.startsWith("/") ||
    normalized.startsWith("./") ||
    normalized.startsWith("../")
  );
}

export function appendMessage(role, text, shouldScroll = true, options = {}) {
  const { attachments = [], messageId = null } = options;
  const hero = document.getElementById("welcome-hero");
  if (hero) hero.remove();

  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;
  if (messageId !== null && messageId !== undefined) {
    msgDiv.dataset.messageId = String(messageId);
  }

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";

  if (role === "ai") {
    const imgMatch = text.match(
      /[sS]aved to:?\s+[`']?.*?\/generated\/([a-zA-Z0-9_-]+\.jpg)[`']?/i,
    );
    if (imgMatch) {
      const filename = imgMatch[1];
      if (!text.includes(`(/generated/${filename})`)) {
        text += `\n\n![Generated Image](/generated/${filename})`;
      }
    }

    msgDiv.dataset.rawContent = text;
    // Safety check for marked
    if (typeof marked !== "undefined") {
      contentDiv.innerHTML = marked.parse(escapeHtml(text));
    } else {
      contentDiv.innerText = text;
    }

    contentDiv.querySelectorAll("a").forEach((a) => {
      const href = a.getAttribute("href") || "";
      if (!isSafeUrl(href)) {
        a.removeAttribute("href");
      } else if (href.startsWith("http://") || href.startsWith("https://")) {
        a.setAttribute("rel", "noopener noreferrer nofollow");
        a.setAttribute("target", "_blank");
      }
    });

    contentDiv.querySelectorAll("img").forEach((img) => {
      const src = img.getAttribute("src") || "";
      if (!isSafeUrl(src, { allowDataImage: true })) {
        img.remove();
      }
    });

    wrapCodeBlocks(contentDiv);
    contentDiv.querySelectorAll("pre code").forEach((block) => {
      if (typeof hljs !== "undefined") hljs.highlightElement(block);
    });

    contentDiv.querySelectorAll("img").forEach((img) => {
      img.style.cursor = "zoom-in";
      img.onclick = () => {
        if (dom.lightboxImg) {
          dom.lightboxImg.src = img.src;
          dom.lightbox.classList.add("active");
        }
      };
    });

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "msg-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.innerHTML =
      '<span class="material-symbols-rounded">content_copy</span> Copy';
    copyBtn.onclick = () => {
      copyToClipboard(text);
    };
    actionsDiv.appendChild(copyBtn);
    contentDiv.appendChild(actionsDiv);
  } else if (role === "system") {
    contentDiv.innerText = text;
  } else {
    msgDiv.dataset.rawContent = text;
    contentDiv.innerHTML = `<p>${escapeHtml(text).replace(/\n/g, "<br>")}</p>`;
    renderAttachments(contentDiv, attachments);

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "msg-actions user-msg-actions";

    const editBtn = document.createElement("button");
    editBtn.className = "msg-action-btn";
    editBtn.innerHTML =
      '<span class="material-symbols-rounded">edit</span> Edit';
    editBtn.onclick = () => {
      if (!userMessageActionHandler) {
        showToast("Message actions are unavailable right now.");
        return;
      }
      userMessageActionHandler("edit", { text, attachments, messageId });
    };
    actionsDiv.appendChild(editBtn);

    const resendBtn = document.createElement("button");
    resendBtn.className = "msg-action-btn";
    resendBtn.innerHTML =
      '<span class="material-symbols-rounded">refresh</span> Resend';
    resendBtn.onclick = () => {
      if (!userMessageActionHandler) {
        showToast("Message actions are unavailable right now.");
        return;
      }
      userMessageActionHandler("resend", { text, attachments, messageId });
    };
    actionsDiv.appendChild(resendBtn);

    contentDiv.appendChild(actionsDiv);
  }

  msgDiv.appendChild(contentDiv);
  if (dom.chatWindow) {
    dom.chatWindow.insertBefore(msgDiv, dom.loadingIndicator);
    if (shouldScroll) dom.chatWindow.scrollTop = dom.chatWindow.scrollHeight;
  }
}
