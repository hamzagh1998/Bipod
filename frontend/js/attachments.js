import { state, dom } from "./state.js";

export function renderAttachmentPreviews() {
  dom.attachmentPreviewContainer.innerHTML = "";

  if (state.currentAttachments.length === 0) {
    dom.attachmentPreviewContainer.classList.remove("active");
    return;
  }

  dom.attachmentPreviewContainer.classList.add("active");

  state.currentAttachments.forEach((attachment, index) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    if (attachment.type === "image") {
      const img = document.createElement("img");
      img.src = `data:image/jpeg;base64,${attachment.content}`;
      item.appendChild(img);
    } else if (attachment.type === "pdf") {
      const icon = document.createElement("div");
      icon.className = "pdf-preview-icon";
      icon.innerHTML = `<span class="material-symbols-rounded">description</span><span class="pdf-name">${attachment.name}</span>`;
      item.appendChild(icon);
    }

    const removeBtn = document.createElement("button");
    removeBtn.className = "preview-remove";
    removeBtn.innerHTML = "&times;";
    removeBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      state.currentAttachments.splice(index, 1);
      renderAttachmentPreviews();
    };
    item.appendChild(removeBtn);

    dom.attachmentPreviewContainer.appendChild(item);
  });
}
