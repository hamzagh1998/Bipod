import { state } from "./state.js";
import { handleLogout } from "./auth.js";

export async function apiFetch(endpoint, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  if (state.authToken) {
    headers["Authorization"] = `Bearer ${state.authToken}`;
  }

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
