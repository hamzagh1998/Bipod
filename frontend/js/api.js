import { state } from "./state.js";
import { handleLogout } from "./auth.js";

function buildHeaders(extraHeaders = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...extraHeaders,
  };

  if (state.authToken) {
    headers.Authorization = `Bearer ${state.authToken}`;
  }

  return headers;
}

export async function apiFetch(endpoint, options = {}) {
  const headers = buildHeaders(options.headers || {});

  try {
    const url = `/api/v1${endpoint}`;
    console.log(`FETCH START [${options.method || "GET"}] ${url}`, {
      headers,
      body: options.body,
    });
    const response = await fetch(url, {
      ...options,
      headers,
    });
    console.log(`FETCH END [${endpoint}] Status:`, response.status);

    if (response.status === 401 && !endpoint.includes("/auth/")) {
      // Unauthorized (except for auth routes themselves)
      handleLogout();
      throw new Error("Session expired. Please login again.");
    }

    return response;
  } catch (err) {
    console.error(`API Fetch Error [${endpoint}]:`, err);
    throw err;
  }
}

export async function apiStream(endpoint, options = {}) {
  const url = `/api/v1${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: buildHeaders(options.headers || {}),
  });

  if (response.status === 401 && !endpoint.includes("/auth/")) {
    handleLogout();
    throw new Error("Session expired. Please login again.");
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      const errorText = await response.text();
      if (errorText) detail = errorText;
    }
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("Streaming is not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  return {
    async *events() {
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            yield JSON.parse(trimmed);
          }
        }

        const trailing = buffer.trim();
        if (trailing) {
          yield JSON.parse(trailing);
        }
      } finally {
        reader.releaseLock();
      }
    },
  };
}
