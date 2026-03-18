import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const STORAGE_KEYS = {
  token: "bipod_token",
  selectedSessionId: "bipod_coach_selected_session_id",
};

const DEFAULT_SESSION_TITLE = "Untitled coaching session";

function safeLocalStorageGet(key, fallback = "") {
  try {
    const value = window.localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

function safeLocalStorageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (error) {
    // Ignore storage failures in private mode or locked-down browsers.
  }
}

function safeLocalStorageRemove(key) {
  try {
    window.localStorage.removeItem(key);
  } catch (error) {
    // Ignore storage failures.
  }
}

function getInitialToken() {
  return (
    safeLocalStorageGet(STORAGE_KEYS.token, "") ||
    safeLocalStorageGet("token", "") ||
    safeLocalStorageGet("auth_token", "")
  );
}

function buildAuthHeaders(token, extra = {}) {
  return {
    Authorization: `Bearer ${token}`,
    ...extra,
  };
}

async function apiFetchJson(path, token, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: buildAuthHeaders(token, options.headers || {}),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(body || `Request failed (${response.status})`);
  }
  return response.json();
}

function makeId(prefix = "coach") {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return `${prefix}_${window.crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function formatDateTime(value) {
  if (!value) {
    return "Just now";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Just now";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function toSummaryTitle(transcript) {
  const words = String(transcript || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (!words.length) {
    return DEFAULT_SESSION_TITLE;
  }
  return words.slice(0, 5).join(" ");
}

function createSession(title = DEFAULT_SESSION_TITLE) {
  const now = new Date().toISOString();
  return {
    id: makeId("session"),
    title,
    target_language: "English",
    native_language: null,
    cefr_level: "A2",
    createdAt: now,
    updatedAt: now,
    turns: [],
  };
}

function createTurn() {
  const now = new Date().toISOString();
  return {
    id: makeId("turn"),
    startedAt: now,
    endedAt: null,
    transcript: "",
    partialTranscript: "",
    reply: "",
    feedback: "",
    score: null,
    fallbackNotices: [],
    events: [],
    error: "",
    status: "recording",
  };
}

function normalizeScore(value) {
  const score = Number.parseFloat(value);
  if (Number.isNaN(score)) {
    return null;
  }
  return score;
}

function eventText(event) {
  return String(
    event?.text ??
      event?.summary ??
      event?.correction ??
      event?.explanation ??
      event?.message ??
      event?.detail ??
      event?.partial ??
      event?.delta ??
      event?.transcript ??
      event?.reply ??
      event?.feedback ??
      event?.value ??
      event?.content ??
      "",
  );
}

function normalizeEventType(value) {
  return String(value || "")
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/[\s-]+/g, "_")
    .toLowerCase();
}

function deriveEventType(rawEvent) {
  const type = normalizeEventType(rawEvent?.type || rawEvent?.event || rawEvent?.kind);
  if (!type && rawEvent?.score !== undefined) {
    return "score";
  }
  if (!type && (rawEvent?.reply || rawEvent?.response)) {
    return "reply";
  }
  if (!type && (rawEvent?.feedback || rawEvent?.coach_feedback)) {
    return "feedback";
  }
  if (!type && (rawEvent?.transcript || rawEvent?.partial_transcript)) {
    return "transcript";
  }
  return type || "message";
}

function updateTurnFromEvent(turn, rawEvent) {
  const eventType = deriveEventType(rawEvent);
  const text = eventText(rawEvent).trim();
  const next = {
    ...turn,
    events: [...turn.events],
  };

  const eventRecord = {
    id: makeId("event"),
    at: new Date().toISOString(),
    type: eventType,
    text,
  };

  if (
    eventType === "partial_transcript" ||
    eventType === "transcript_partial" ||
    eventType === "speech_partial" ||
    eventType === "stt_partial"
  ) {
    next.partialTranscript = text || eventText(rawEvent);
    next.status = "streaming";
    next.events.push({ ...eventRecord, label: "Partial transcript" });
    return next;
  }

  if (
    eventType === "transcript" ||
    eventType === "final_transcript" ||
    eventType === "speech_final" ||
    eventType === "transcript_final" ||
    eventType === "stt_final"
  ) {
    next.transcript = text || next.transcript;
    next.partialTranscript = "";
    next.status = "streaming";
    next.events.push({ ...eventRecord, label: "Transcript" });
    return next;
  }

  if (eventType === "reply" || eventType === "assistant" || eventType === "coach_reply" || eventType === "response") {
    next.reply = text || next.reply;
    next.events.push({ ...eventRecord, label: "Coach reply" });
    return next;
  }

  if (eventType === "feedback" || eventType === "coach_feedback") {
    next.feedback = text || next.feedback;
    next.events.push({ ...eventRecord, label: "Feedback" });
    return next;
  }

  if (eventType === "score") {
    next.score = normalizeScore(rawEvent?.score ?? rawEvent?.value ?? rawEvent?.rating ?? text);
    next.events.push({ ...eventRecord, label: "Score" });
    return next;
  }

  if (eventType === "model_fallback" || eventType === "fallback_notice" || eventType === "fallback" || eventType === "notice" || eventType === "warning") {
    next.fallbackNotices = [...next.fallbackNotices, text || "Fallback path used"];
    next.events.push({ ...eventRecord, label: "Fallback notice" });
    return next;
  }

  if (eventType === "status" || eventType === "progress" || eventType === "stage") {
    next.events.push({ ...eventRecord, label: "Status" });
    return next;
  }

  if (eventType === "error") {
    next.error = text || "Coach stream failed";
    next.status = "error";
    next.events.push({ ...eventRecord, label: "Error" });
    return next;
  }

  next.events.push({ ...eventRecord, label: eventType.replace(/_/g, " ") || "Event" });
  return next;
}

function summarizeTurns(turns) {
  const count = turns.length;
  const scores = turns
    .map((turn) => Number(turn.score))
    .filter((value) => !Number.isNaN(value));
  const averageScore = scores.length
    ? scores.reduce((sum, value) => sum + value, 0) / scores.length
    : null;
  const transcriptCount = turns.filter((turn) => turn.transcript || turn.partialTranscript).length;
  const fallbackCount = turns.reduce((sum, turn) => sum + (turn.fallbackNotices?.length || 0), 0);
  const latestTurn = turns[0] || null;

  return {
    count,
    averageScore,
    transcriptCount,
    fallbackCount,
    latestTurn,
    completionRate: count ? Math.round((transcriptCount / count) * 100) : 0,
  };
}

function SessionList({ sessions, currentSessionId, onSelectSession, onCreateSession }) {
  return (
    <div className="coach-sidebar-section">
      <div className="coach-action-row">
        <button type="button" className="coach-btn primary" onClick={onCreateSession}>
          <span className="material-symbols-rounded">add</span>
          New session
        </button>
        <button
          type="button"
          className="coach-btn secondary"
          onClick={() => window.location.assign("/")}
          title="Back to the existing chat app"
        >
          <span className="material-symbols-rounded">chat</span>
        </button>
      </div>

      <div className="coach-section-title">Sessions</div>
      <div className="coach-session-list coach-scrollbar">
        {sessions.length === 0 ? (
          <div className="coach-empty-state">
            <h3>No sessions yet</h3>
            <p>Create a session to start tracking coaching turns.</p>
          </div>
        ) : (
          sessions.map((session) => (
            <button
              key={session.id}
              type="button"
              className={`coach-session-item ${session.id === currentSessionId ? "active" : ""}`}
              onClick={() => onSelectSession(session.id)}
            >
              <div className="coach-session-title">
                <span>{session.title || DEFAULT_SESSION_TITLE}</span>
                <span className="coach-muted">{session.turns?.length || 0}</span>
              </div>
              <div className="coach-session-meta">
                <span>{formatDateTime(session.updatedAt)}</span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

function SummaryPanel({ summary }) {
  const scoreLabel = summary.averageScore == null ? "No score yet" : summary.averageScore.toFixed(1);
  const scoreProgress = summary.averageScore == null ? 0 : Math.max(0, Math.min(100, summary.averageScore));

  return (
    <div className="coach-summary-grid">
      <div className="coach-summary-card">
        <p className="coach-summary-label">Turns</p>
        <p className="coach-summary-value">{summary.count}</p>
        <p className="coach-summary-copy">{summary.transcriptCount} with transcript output</p>
      </div>
      <div className="coach-summary-card">
        <p className="coach-summary-label">Average score</p>
        <p className="coach-summary-value">{scoreLabel}</p>
        <p className="coach-summary-copy">Calculated from scored turns only</p>
        <div className="coach-progress-bar" aria-hidden="true">
          <span style={{ width: `${scoreProgress}%` }} />
        </div>
      </div>
      <div className="coach-summary-card">
        <p className="coach-summary-label">Fallback notices</p>
        <p className="coach-summary-value">{summary.fallbackCount}</p>
        <p className="coach-summary-copy">Model or pipeline fallback events</p>
      </div>
      <div className="coach-summary-card">
        <p className="coach-summary-label">Completion</p>
        <p className="coach-summary-value">{summary.completionRate}%</p>
        <p className="coach-summary-copy">Turns with recorded transcript output</p>
      </div>
    </div>
  );
}

function LiveTurnCard({ turn, isLive, recordingDuration }) {
  const displayTranscript = turn.transcript || turn.partialTranscript;
  const scoreLabel = turn.score == null ? "Pending score" : turn.score.toFixed(1);

  return (
    <div className="coach-live-card">
      <div className="coach-live-top">
        <div>
          <h3>Live turn</h3>
          <div className="coach-subtle">
            {isLive ? "Recording and streaming audio to the coach backend." : "Ready for the next turn."}
          </div>
        </div>
        <div className="coach-status">
          <span className="material-symbols-rounded">{isLive ? "mic" : "graphic_eq"}</span>
          <strong>{isLive ? `Recording ${recordingDuration}` : "Idle"}</strong>
        </div>
      </div>

      <div className="coach-live-transcript coach-scrollbar">
        {displayTranscript ? (
          <>
            <div>{turn.transcript || turn.partialTranscript}</div>
            {turn.partialTranscript && !turn.transcript ? <div className="partial">Listening for the final transcript...</div> : null}
          </>
        ) : (
          <div className="coach-muted">
            Hold the button below and speak. Partial transcript, reply, feedback, and score events will appear here.
          </div>
        )}
      </div>

      <div className="coach-turn-badges" style={{ marginTop: "14px" }}>
        <span className="coach-badge">
          <span className="material-symbols-rounded">schedule</span>
          <strong>{formatDateTime(turn.startedAt)}</strong>
        </span>
        <span className="coach-badge">
          <span className="material-symbols-rounded">star</span>
          <strong>{scoreLabel}</strong>
        </span>
        <span className="coach-badge">
          <span className="material-symbols-rounded">warning</span>
          <strong>{turn.fallbackNotices.length}</strong>
        </span>
      </div>
    </div>
  );
}

function TurnCard({ turn, index }) {
  return (
    <article className="coach-turn-card">
      <div className="coach-turn-header">
        <h4>Turn {index + 1}</h4>
        <time dateTime={turn.endedAt || turn.startedAt}>{formatDateTime(turn.endedAt || turn.startedAt)}</time>
      </div>

      <div className="coach-turn-badges">
        <span className="coach-badge">
          <span className="material-symbols-rounded">short_text</span>
          <strong>{turn.transcript ? "Final transcript" : "No final transcript"}</strong>
        </span>
        <span className="coach-badge">
          <span className="material-symbols-rounded">chat_bubble</span>
          <strong>{turn.reply ? "Reply ready" : "Reply pending"}</strong>
        </span>
        <span className="coach-badge">
          <span className="material-symbols-rounded">reviews</span>
          <strong>{turn.feedback ? "Feedback ready" : "Feedback pending"}</strong>
        </span>
        <span className="coach-badge">
          <span className="material-symbols-rounded">star</span>
          <strong>{turn.score == null ? "No score" : turn.score.toFixed(1)}</strong>
        </span>
      </div>

      <div className="coach-event-log">
        {turn.transcript ? (
          <div className="coach-event partial">
            <div className="label">Transcript</div>
            <div className="text">{turn.transcript}</div>
          </div>
        ) : null}
        {!turn.transcript && turn.partialTranscript ? (
          <div className="coach-event partial">
            <div className="label">Partial transcript</div>
            <div className="text">{turn.partialTranscript}</div>
          </div>
        ) : null}
        {turn.reply ? (
          <div className="coach-event reply">
            <div className="label">Coach reply</div>
            <div className="text">{turn.reply}</div>
          </div>
        ) : null}
        {turn.feedback ? (
          <div className="coach-event feedback">
            <div className="label">Feedback</div>
            <div className="text">{turn.feedback}</div>
          </div>
        ) : null}
        {turn.fallbackNotices.map((notice, idx) => (
          <div className="coach-event" key={`${turn.id}-notice-${idx}`}>
            <div className="label">Fallback notice</div>
            <div className="text">{notice}</div>
          </div>
        ))}
        {turn.error ? (
          <div className="coach-event error">
            <div className="label">Error</div>
            <div className="text">{turn.error}</div>
          </div>
        ) : null}
        {turn.events.slice(-4).map((event) => (
          <div className={`coach-event ${event.type === "reply" ? "reply" : event.type === "feedback" ? "feedback" : event.type === "error" ? "error" : ""}`} key={event.id}>
            <div className="label">{event.label || event.type}</div>
            <div className="text">{event.text || "Event received"}</div>
          </div>
        ))}
      </div>
    </article>
  );
}

function AuthGate() {
  return (
    <div className="coach-auth-screen">
      <div className="coach-auth-card">
        <div className="coach-brand">
          <img src="/static/logo.png" alt="Bipod" />
          <div>
            <h1>Bipod Coach</h1>
            <p>Practice turns, get feedback, and review progress.</p>
          </div>
        </div>
        <h2>Sign in to continue</h2>
        <p>
          This page reuses the existing Bipod auth token from <code>bipod_token</code>. Log in through the main app first,
          then return here to start a coaching session.
        </p>
        <div className="coach-auth-actions">
          <a className="coach-btn primary" href="/">
            <span className="material-symbols-rounded">login</span>
            Go to login
          </a>
          <a className="coach-btn secondary" href="/studio.html">
            <span className="material-symbols-rounded">palette</span>
            Open studio
          </a>
        </div>
      </div>
    </div>
  );
}

function CoachApp() {
  const [token, setToken] = useState(() => getInitialToken());
  const [authState, setAuthState] = useState(token ? "checking" : "missing");
  const [username, setUsername] = useState("");
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(() => safeLocalStorageGet(STORAGE_KEYS.selectedSessionId, ""));
  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState("00:00");
  const [errorMessage, setErrorMessage] = useState("");
  const [statusMessage, setStatusMessage] = useState("Ready");
  const [currentTurnId, setCurrentTurnId] = useState(null);
  const [liveTurnVersion, setLiveTurnVersion] = useState(0);

  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const chunksRef = useRef([]);
  const recorderStartRef = useRef(0);
  const durationIntervalRef = useRef(null);
  const streamAbortRef = useRef(null);
  const currentTurnIdRef = useRef(null);
  const currentSessionIdRef = useRef(currentSessionId);
  const sessionsRef = useRef(sessions);
  const isStoppingRef = useRef(false);

  const currentSession = useMemo(
    () => sessions.find((session) => session.id === currentSessionId) || sessions[0] || null,
    [sessions, currentSessionId],
  );

  const currentTurn = useMemo(() => {
    if (!currentSession) {
      return createTurn();
    }
    return currentSession.turns.find((turn) => turn.id === currentTurnId) || currentSession.turns[0] || createTurn();
  }, [currentSession, currentTurnId, liveTurnVersion]);

  const summary = useMemo(() => summarizeTurns(currentSession?.turns || []), [currentSession, liveTurnVersion]);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
    safeLocalStorageSet(STORAGE_KEYS.selectedSessionId, currentSessionId);
  }, [currentSessionId]);

  useEffect(() => {
    if (token) {
      safeLocalStorageSet(STORAGE_KEYS.token, token);
    } else {
      safeLocalStorageRemove(STORAGE_KEYS.token);
    }
  }, [token]);

  useEffect(() => {
    if (!token) {
      setAuthState("missing");
      return;
    }

    let cancelled = false;

    async function loadSessionTurns(sessionId) {
      const turns = await apiFetchJson(`/api/v1/coach/sessions/${sessionId}/turns`, token);
      setSessions((prevSessions) =>
        prevSessions.map((session) =>
          session.id !== sessionId
            ? session
            : {
                ...session,
                turns: turns.map((turn) => ({
                  id: turn.id,
                  startedAt: turn.created_at,
                  endedAt: turn.created_at,
                  transcript: turn.transcript || "",
                  partialTranscript: "",
                  reply: turn.reply || "",
                  feedback: turn.explanation || "",
                  score: turn.score,
                  fallbackNotices: [],
                  events: [],
                  error: "",
                  status: "done",
                })),
              },
        ),
      );
    }

    async function hydrateSessions() {
      const apiSessions = await apiFetchJson("/api/v1/coach/sessions", token);
      const normalized = apiSessions.map((session) => ({
        id: session.id,
        title: session.title || DEFAULT_SESSION_TITLE,
        target_language: session.target_language || "English",
        native_language: session.native_language || null,
        cefr_level: session.cefr_level || "A2",
        createdAt: session.created_at,
        updatedAt: session.updated_at,
        turns: [],
      }));
      if (normalized.length === 0) {
        const created = await apiFetchJson("/api/v1/coach/sessions", token, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: DEFAULT_SESSION_TITLE, target_language: "English", cefr_level: "A2" }),
        });
        normalized.push({
          id: created.id,
          title: created.title || DEFAULT_SESSION_TITLE,
          target_language: created.target_language || "English",
          native_language: created.native_language || null,
          cefr_level: created.cefr_level || "A2",
          createdAt: created.created_at,
          updatedAt: created.updated_at,
          turns: [],
        });
      }

      setSessions(normalized);
      const savedId = safeLocalStorageGet(STORAGE_KEYS.selectedSessionId, "");
      const chosenId = normalized.some((session) => session.id === savedId) ? savedId : normalized[0].id;
      setCurrentSessionId(chosenId);
      await loadSessionTurns(chosenId);
    }

    async function fetchIdentity() {
      setAuthState("checking");
      try {
        const response = await fetch("/api/v1/auth/me", { headers: buildAuthHeaders(token) });

        if (!response.ok) {
          if (!cancelled) {
            if (response.status === 401) {
              setAuthState("invalid");
              setStatusMessage("Token rejected by the API");
              safeLocalStorageRemove(STORAGE_KEYS.token);
              setToken("");
            } else {
              // Non-auth failures shouldn't lock a user out of coach UI.
              setAuthState("ready");
              setUsername("Coach user");
              setStatusMessage("Ready");
              await hydrateSessions();
            }
          }
          return;
        }

        const payload = await response.json();
        if (!cancelled) {
          setUsername(payload.username || "Coach user");
          setAuthState("ready");
          setStatusMessage("Ready");
          await hydrateSessions();
        }
      } catch (error) {
        if (!cancelled) {
          // Network/backend hiccups shouldn't force sign-in gate.
          setAuthState("ready");
          setUsername("Coach user");
          setStatusMessage("Ready");
          try {
            await hydrateSessions();
          } catch {
            // Keep page usable even if session hydrate fails.
          }
        }
      }
    }

    fetchIdentity();

    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    if (!sessions.some((session) => session.id === currentSessionId) && sessions[0]) {
      setCurrentSessionId(sessions[0].id);
    }
  }, [sessions, currentSessionId]);

  useEffect(() => {
    return () => {
      stopRecording(true);
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function updateSession(sessionId, updater) {
    setSessions((prevSessions) =>
      prevSessions.map((session) =>
        session.id === sessionId ? updater(session) : session,
      ),
    );
  }

  async function createNewSession() {
    if (!token) {
      return;
    }
    try {
      const created = await apiFetchJson("/api/v1/coach/sessions", token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: DEFAULT_SESSION_TITLE, target_language: "English", cefr_level: "A2" }),
      });
      const session = {
        id: created.id,
        title: created.title || DEFAULT_SESSION_TITLE,
        target_language: created.target_language || "English",
        native_language: created.native_language || null,
        cefr_level: created.cefr_level || "A2",
        createdAt: created.created_at,
        updatedAt: created.updated_at,
        turns: [],
      };
      setSessions((prevSessions) => [session, ...prevSessions].slice(0, 20));
      setCurrentSessionId(session.id);
      setCurrentTurnId(null);
      setErrorMessage("");
      setStatusMessage("Created a new session");
    } catch (error) {
      setErrorMessage(error?.message || "Could not create session.");
    }
  }

  async function selectSession(sessionId) {
    setCurrentSessionId(sessionId);
    const session = sessionsRef.current.find((item) => item.id === sessionId);
    setCurrentTurnId(session?.turns?.[0]?.id || null);
    setErrorMessage("");
    if (token) {
      try {
        const turns = await apiFetchJson(`/api/v1/coach/sessions/${sessionId}/turns`, token);
        setSessions((prevSessions) =>
          prevSessions.map((item) =>
            item.id !== sessionId
              ? item
              : {
                  ...item,
                  turns: turns.map((turn) => ({
                    id: turn.id,
                    startedAt: turn.created_at,
                    endedAt: turn.created_at,
                    transcript: turn.transcript || "",
                    partialTranscript: "",
                    reply: turn.reply || "",
                    feedback: turn.explanation || "",
                    score: turn.score,
                    fallbackNotices: [],
                    events: [],
                    error: "",
                    status: "done",
                  })),
                },
          ),
        );
      } catch (error) {
        setErrorMessage(error?.message || "Could not load turns.");
      }
    }
  }

  function beginTurn() {
    if (!currentSession) {
      createNewSession();
      return;
    }

    const turn = createTurn();
    currentTurnIdRef.current = turn.id;
    setCurrentTurnId(turn.id);
    setErrorMessage("");
    setStatusMessage("Listening");

    updateSession(currentSession.id, (session) => ({
      ...session,
      updatedAt: new Date().toISOString(),
      turns: [turn, ...(session.turns || [])],
      title: session.title,
    }));

    setLiveTurnVersion((value) => value + 1);
    return turn;
  }

  async function startRecording() {
    if (isRecording || isSending) {
      return;
    }

    if (!token) {
      setAuthState("missing");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setErrorMessage("This browser does not support audio capture.");
      return;
    }

    try {
      const sessionId = currentSessionIdRef.current || currentSession?.id;
      if (!sessionId) {
        createNewSession();
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
      const mimeType = mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || "";
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;
      mediaStreamRef.current = stream;
      chunksRef.current = [];
      recorderStartRef.current = Date.now();
      isStoppingRef.current = false;

      const turn = beginTurn();
      if (!turn) {
        return;
      }

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setErrorMessage("Recorder error. Please try again.");
      };

      recorder.onstop = async () => {
        const nextBlob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        releaseStream();
        setIsRecording(false);
        if (!isStoppingRef.current) {
          await sendAudioTurn(nextBlob);
        }
      };

      recorder.start(250);
      setIsRecording(true);
      setRecordingDuration("00:00");
      durationIntervalRef.current = window.setInterval(() => {
        setRecordingDuration(formatDuration(Date.now() - recorderStartRef.current));
      }, 250);
    } catch (error) {
      setErrorMessage(error?.message || "Microphone access failed.");
      releaseStream();
      setIsRecording(false);
    }
  }

  function releaseStream() {
    if (durationIntervalRef.current) {
      window.clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    mediaRecorderRef.current = null;
  }

  function stopRecording(force = false) {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === "inactive") {
      releaseStream();
      setIsRecording(false);
      return;
    }

    isStoppingRef.current = force;
    try {
      mediaRecorderRef.current.stop();
    } catch (error) {
      releaseStream();
      setIsRecording(false);
    }
  }

  function attachEventToCurrentTurn(rawEvent) {
    const sessionId = currentSessionIdRef.current;
    const turnId = currentTurnIdRef.current;

    if (!sessionId || !turnId) {
      return;
    }

    updateSession(sessionId, (session) => {
      const turns = (session.turns || []).map((turn) => {
        if (turn.id !== turnId) {
          return turn;
        }
        return updateTurnFromEvent(turn, rawEvent);
      });

      const nextSession = {
        ...session,
        turns,
        updatedAt: new Date().toISOString(),
      };

      const activeTurn = turns.find((turn) => turn.id === turnId);
      if (activeTurn?.transcript && nextSession.title === DEFAULT_SESSION_TITLE) {
        nextSession.title = toSummaryTitle(activeTurn.transcript);
      }

      return nextSession;
    });

    setLiveTurnVersion((value) => value + 1);
  }

  async function sendAudioTurn(audioBlob) {
    const sessionId = currentSessionIdRef.current;
    const turnId = currentTurnIdRef.current;
    if (!sessionId || !turnId) {
      setStatusMessage("Session unavailable");
      return;
    }

    setIsSending(true);
    setStatusMessage("Uploading and streaming response");
    setErrorMessage("");

    const controller = new AbortController();
    streamAbortRef.current = controller;

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, `coach-${turnId}.webm`);
      formData.append("file", audioBlob, `coach-${turnId}.webm`);
      formData.append("session_id", sessionId);
      formData.append("session_title", currentSession?.title || DEFAULT_SESSION_TITLE);
      formData.append("turn_id", turnId);
      formData.append("client_timestamp", new Date().toISOString());

      const response = await fetch("/api/v1/coach/turns/stream", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const text = await response.text().catch(() => "");
        throw new Error(text || `Coach stream failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamComplete = false;

      while (!streamComplete) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        let newlineIndex = buffer.indexOf("\n");
        while (newlineIndex >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          newlineIndex = buffer.indexOf("\n");

          if (!line) {
            continue;
          }

          let event;
          try {
            event = JSON.parse(line);
          } catch (parseError) {
            continue;
          }

          attachEventToCurrentTurn(event);

          if (String(event.type || "").toLowerCase() === "error") {
            throw new Error(eventText(event) || "Coach stream reported an error");
          }

          if (String(event.type || "").toLowerCase() === "done") {
            streamComplete = true;
            break;
          }
        }
      }

      updateSession(sessionId, (session) => {
        const turns = (session.turns || []).map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                endedAt: new Date().toISOString(),
                status: "done",
              }
            : turn,
        );
        return {
          ...session,
          turns,
          updatedAt: new Date().toISOString(),
        };
      });

      setStatusMessage("Turn completed");
    } catch (error) {
      if (error.name !== "AbortError") {
        setErrorMessage(error?.message || "Failed to stream the turn.");
        setStatusMessage("Turn failed");
        attachEventToCurrentTurn({
          type: "error",
          detail: error?.message || "Failed to stream the turn.",
        });
      }
    } finally {
      setIsSending(false);
      streamAbortRef.current = null;
      currentTurnIdRef.current = null;
    }
  }

  function handlePttPointerDown(event) {
    event.preventDefault();
    if (!isRecording) {
      startRecording();
    }
  }

  function handlePttPointerUp(event) {
    event.preventDefault();
    if (isRecording) {
      stopRecording();
    }
  }

  function handlePttKeyDown(event) {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      if (!isRecording) {
        startRecording();
      }
    }
  }

  function handlePttKeyUp(event) {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      if (isRecording) {
        stopRecording();
      }
    }
  }

  const activeTurn = currentTurn && currentTurn.id === currentTurnId ? currentTurn : currentSession?.turns?.[0] || createTurn();

  if (authState === "missing" || authState === "invalid") {
    return <AuthGate />;
  }

  return (
    <div className="coach-app">
      <aside className="coach-sidebar">
        <div className="coach-brand">
          <img src="/static/logo.png" alt="Bipod" />
          <div>
            <h1>Bipod Coach</h1>
            <p>Live voice coaching and session review</p>
          </div>
        </div>

        <div className="coach-pill">
          <span className="material-symbols-rounded">account_circle</span>
          <span>{username || "Authenticating..."}</span>
        </div>

        <SessionList
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={selectSession}
          onCreateSession={createNewSession}
        />
      </aside>

      <main className="coach-main">
        <header className="coach-header">
          <div className="coach-heading">
            <div className="coach-pill" style={{ marginBottom: "6px" }}>
              <span className="material-symbols-rounded">graphic_eq</span>
              <span>{statusMessage}</span>
            </div>
            <h2>{currentSession?.title || DEFAULT_SESSION_TITLE}</h2>
            <p>
              Record a response with push-to-talk. The coach stream will surface partial transcript, reply, feedback, score,
              and fallback notices as events arrive.
            </p>
          </div>

          <div className="coach-status-stack">
            <div className="coach-status">
              <span className="material-symbols-rounded">event</span>
              <span>
                <strong>{sessions.length}</strong> sessions
              </span>
            </div>
            <div className="coach-status">
              <span className="material-symbols-rounded">mic</span>
              <span>
                <strong>{summary.count}</strong> turns
              </span>
            </div>
            <div className="coach-status">
              <span className="material-symbols-rounded">schedule</span>
              <span>
                <strong>{currentSession ? formatDateTime(currentSession.updatedAt) : "Just now"}</strong>
              </span>
            </div>
          </div>
        </header>

        <SummaryPanel summary={summary} />

        <section className="coach-workspace">
          <LiveTurnCard turn={activeTurn} isLive={isRecording || isSending} recordingDuration={recordingDuration} />

          <div className="coach-turn-list coach-scrollbar">
            {currentSession?.turns?.length ? (
              currentSession.turns.map((turn, index) => <TurnCard key={turn.id} turn={turn} index={index} />)
            ) : (
              <div className="coach-empty-state">
                <h3>No recorded turns yet</h3>
                <p>Press and hold the button below to record your first coaching turn.</p>
              </div>
            )}
          </div>
        </section>

        <footer className="coach-footer">
          <button
            type="button"
            className={`coach-ptt ${isRecording ? "recording" : ""}`}
            onPointerDown={handlePttPointerDown}
            onPointerUp={handlePttPointerUp}
            onPointerCancel={handlePttPointerUp}
            onPointerLeave={handlePttPointerUp}
            onKeyDown={handlePttKeyDown}
            onKeyUp={handlePttKeyUp}
            aria-pressed={isRecording}
            disabled={isSending || authState !== "ready"}
          >
            <span className="material-symbols-rounded">{isRecording ? "stop_circle" : "mic"}</span>
            {isRecording ? "Release to send" : "Push to talk"}
          </button>

          <div className="coach-ptt-hint">
            {errorMessage ? <div style={{ color: "var(--coach-danger)", marginBottom: "6px" }}>{errorMessage}</div> : null}
            <div>
              Hold the button while speaking. Release to upload audio to <code>/api/v1/coach/turns/stream</code> and watch the
              streamed coaching events appear above.
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}

const root = createRoot(document.getElementById("coach-root"));
root.render(<CoachApp />);
