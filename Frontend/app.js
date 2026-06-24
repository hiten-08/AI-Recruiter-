// =============================================================================
// app.js  —  Shared utilities for all pages
// =============================================================================
//
// This file is loaded by index.html, submit.html, and results.html.
// It handles:
//   1. Toast notifications (small pop-up messages)
//   2. The submitJD() function — calls the backend API and stores results
//   3. Shared helpers
//
// HOW THE FRONTEND ↔ BACKEND FLOW WORKS:
//   submit.html  →  calls submitJD()
//                →  POST /rank to the FastAPI backend
//                →  stores response in localStorage
//                →  redirects to results.html
//   results.html →  reads localStorage
//                →  renders the candidate cards
// =============================================================================

// Change this if your backend runs on a different port
const API_BASE = "http://localhost:8000";

// =============================================================================
// TOAST — small notification that appears bottom-right
// =============================================================================
function showToast(message, duration = 3000) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), duration);
}

// =============================================================================
// SUBMIT JD — called from submit.html's form submit handler
// =============================================================================
async function submitJD({ title, jdText, location, experience }) {
  const btn = document.getElementById("submitBtn");
  const loadingOverlay = document.getElementById("loadingOverlay");

  // --- Show loading state ---
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Ranking…';
  }
  if (loadingOverlay) {
    loadingOverlay.classList.add("show");
  }

  try {
    // --- Check if backend is ready ---
    let healthResp;
    try {
      healthResp = await fetch(`${API_BASE}/health`);
    } catch (err) {
      throw new Error(
        "Cannot connect to the backend. Make sure the API server is running on port 8000.\n\nRun: uvicorn api:app --port 8000"
      );
    }

    const health = await healthResp.json();
    if (health.status === "initializing") {
      throw new Error(
        "The server is still loading candidates. Please wait 30 seconds and try again."
      );
    }

    // --- Build the full JD text ---
    // Include location + experience range in the JD text if provided,
    // so the ranking engine picks them up in its parsing
    let fullJD = jdText;
    if (location) fullJD += `\n\nPreferred Location: ${location}`;
    if (experience) fullJD += `\nExperience Required: ${experience}`;

    // --- Call the ranking API ---
    const response = await fetch(`${API_BASE}/rank`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jd: fullJD, top_n: 100 }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();

    // --- Store results in localStorage for results.html to read ---
    localStorage.setItem("redrob_results", JSON.stringify(data.results));
    localStorage.setItem(
      "redrob_meta",
      JSON.stringify({
        title,
        submittedAt: new Date().toLocaleString("en-IN", {
          dateStyle: "medium",
          timeStyle: "short",
        }),
        elapsedSeconds: data.elapsed_seconds,
        totalRanked: data.total_ranked,
      })
    );

    // --- Redirect to results page ---
    window.location.href = "results.html";

  } catch (err) {
    // --- Hide loading, show error ---
    if (loadingOverlay) loadingOverlay.classList.remove("show");
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "Show Results →";
    }
    showToast("Error: " + err.message, 6000);
    console.error("submitJD error:", err);
  }
}

// =============================================================================
// HEALTH POLL — used on results.html to show a "connecting..." state
// while the backend loads (useful on first startup)
// =============================================================================
async function waitForBackend(onReady, onStatus) {
  const maxAttempts = 20;
  let attempts = 0;

  const poll = async () => {
    attempts++;
    try {
      const resp = await fetch(`${API_BASE}/health`);
      const data = await resp.json();

      if (data.status === "ready") {
        onReady(data);
      } else if (attempts < maxAttempts) {
        if (onStatus) onStatus(`Loading ${data.candidates?.toLocaleString() || "..."} candidates...`);
        setTimeout(poll, 2000);
      } else {
        if (onStatus) onStatus("Backend taking longer than expected...");
      }
    } catch {
      if (attempts < maxAttempts) {
        if (onStatus) onStatus("Connecting to backend...");
        setTimeout(poll, 2000);
      }
    }
  };

  poll();
}
