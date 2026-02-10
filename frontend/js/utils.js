// --- Toast Notification ---
export function showToast(message, duration = 2000) {
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

// --- Clipboard Helper ---
export async function copyToClipboard(text) {
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

// --- Markdown Configuration ---
export function setupMarkdown() {
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
}

// --- Hero & UI Helpers ---
export function createWelcomeHero() {
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

export function wrapCodeBlocks(container) {
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
