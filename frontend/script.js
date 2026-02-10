const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

function appendMessage(role, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;

  // Simple markdown-ish rendering for code blocks
  const formattedText = text.replace(
    /```([\s\S]*?)```/g,
    "<pre><code>$1</code></pre>",
  );
  msgDiv.innerHTML = `<p>${formattedText.replace(/\n/g, "<br>")}</p>`;

  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  // Clear input and show user message
  userInput.value = "";
  appendMessage("user", text);

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    if (!response.ok) throw new Error("Brain connection lost");

    const data = await response.json();
    appendMessage("ai", data.response);
  } catch (error) {
    appendMessage("system", `Error: ${error.message}`);
  }
});
