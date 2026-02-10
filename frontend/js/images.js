import { state, dom } from "./state.js";

export function renderImagePreviews() {
  dom.imagePreviewContainer.innerHTML = "";

  if (state.currentImages.length === 0) {
    dom.imagePreviewContainer.classList.remove("active");
    return;
  }

  dom.imagePreviewContainer.classList.add("active");

  state.currentImages.forEach((b64, index) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${b64}`;

    item.appendChild(img);

    const removeBtn = document.createElement("button");
    removeBtn.className = "preview-remove";
    removeBtn.innerHTML = "&times;";
    removeBtn.onclick = () => {
      state.currentImages.splice(index, 1);
      renderImagePreviews();
    };
    item.appendChild(removeBtn);

    dom.imagePreviewContainer.appendChild(item);
  });
}
