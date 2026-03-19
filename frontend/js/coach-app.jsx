import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const STORAGE_KEYS = {
  token: "bipod_token",
};

const SUBJECT_OPTIONS = ["Job interview", "Travel", "Technology", "Daily life", "Business meeting", "Custom"];

const LANGUAGE_OPTIONS = [
  "English",
  "Arabic",
  "French",
  "Spanish",
  "German",
  "Italian",
  "Portuguese",
  "Russian",
  "Turkish",
  "Hindi",
  "Urdu",
  "Chinese",
  "Japanese",
  "Korean",
  "Dutch",
  "Swedish",
  "Polish",
  "Ukrainian",
  "Greek",
];

const PERSONA_OPTIONS = [
  {
    id: "default_coach",
    label: "Default coach",
    prompt:
      "Persona: Warm but strict language coach. Keep replies concise, clear, and practical. Ask one focused follow-up question each turn.",
  },
  {
    id: "anby",
    label: "Anby",
    prompt:
      "Persona: Calm, tactical, low-drama, mission-focused speaker with short, precise sentences, dry wit, and practical advice. Stay helpful and respectful.",
  },
  {
    id: "friendly_mentor",
    label: "Friendly mentor",
    prompt:
      "Persona: Supportive mentor tone. Encouraging and clear, but still corrective. Keep guidance simple and actionable.",
  },
  {
    id: "bmo",
    label: "BMO",
    prompt:
      "Persona: Playful, cheerful, and curious. Keep replies short and upbeat, ask simple follow-up questions, and stay supportive.",
  },
  {
    id: "goku",
    label: "Goku",
    prompt:
      "Persona: Energetic, straightforward, and positive. Encourage effort, keep language simple, and push for one stronger sentence each turn.",
  },
  {
    id: "gute",
    label: "Gute",
    prompt:
      "Persona: Calm, polite, and precise. Give practical corrections and ask one clear question at a time.",
  },
];

const VOICE_TO_PERSONA_ID = {
  "builtin:anby": "anby",
  "builtin:bmo": "bmo",
  "builtin:goku": "goku",
  "builtin:gute": "gute",
};

const LANGUAGE_TO_BCP47 = {
  English: "en-US",
  Arabic: "ar-SA",
  French: "fr-FR",
  Spanish: "es-ES",
  German: "de-DE",
  Italian: "it-IT",
  Portuguese: "pt-PT",
  Russian: "ru-RU",
  Turkish: "tr-TR",
  Hindi: "hi-IN",
  Urdu: "ur-PK",
  Chinese: "zh-CN",
  Japanese: "ja-JP",
  Korean: "ko-KR",
  Dutch: "nl-NL",
  Swedish: "sv-SE",
  Polish: "pl-PL",
  Ukrainian: "uk-UA",
  Greek: "el-GR",
};

const SERVER_VOICE_OPTIONS = [
  { id: "server:default", label: "Local default" },
  { id: "server:female", label: "Local female" },
  { id: "server:male", label: "Local male" },
];

const DEFAULT_SERVER_VOICE = SERVER_VOICE_OPTIONS[0].id;
const DEFAULT_BUILTIN_VOICE_CHOICE = "builtin:anby";
const DEFAULT_PROFILE_NAME = "My cloned voice";

function parseVoiceChoice(value) {
  const raw = String(value || "");
  if (raw === "session_clone") {
    return {
      voiceMode: "cloned_session",
      voicePreset: "default",
      voiceProfileId: "",
      builtinVoiceId: "",
      useBrowserVoice: false,
    };
  }
  if (raw.startsWith("server:")) {
    return {
      voiceMode: "preset",
      voicePreset: raw.slice("server:".length) || "default",
      voiceProfileId: "",
      builtinVoiceId: "",
      useBrowserVoice: false,
    };
  }
  if (raw.startsWith("builtin:")) {
    return {
      voiceMode: "preset",
      voicePreset: "default",
      voiceProfileId: "",
      builtinVoiceId: raw.slice("builtin:".length),
      useBrowserVoice: false,
    };
  }
  if (raw.startsWith("profile:")) {
    return {
      voiceMode: "cloned_profile",
      voicePreset: "default",
      voiceProfileId: raw.slice("profile:".length),
      builtinVoiceId: "",
      useBrowserVoice: false,
    };
  }
  return {
    voiceMode: "preset",
    voicePreset: "default",
    voiceProfileId: "",
    builtinVoiceId: "",
    useBrowserVoice: true,
  };
}

function safeLocalStorageGet(key, fallback = "") {
  try {
    const value = window.localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch {
    return fallback;
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
    let detail = body || `Request failed (${response.status})`;
    if (body) {
      try {
        const parsed = JSON.parse(body);
        if (parsed && typeof parsed === "object" && typeof parsed.detail === "string" && parsed.detail.trim()) {
          detail = parsed.detail.trim();
        }
      } catch {
        // keep raw body when not JSON
      }
    }
    throw new Error(detail);
  }
  return response.json();
}

function makeId(prefix = "coach") {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return `${prefix}_${window.crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function normalizeMistakes(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => ({
      category: String(item?.category || "general"),
      detail: String(item?.detail || "").trim(),
      severity: String(item?.severity || "medium"),
      suggestion: String(item?.suggestion || "").trim(),
    }))
    .filter((item) => item.detail || item.suggestion);
}

function eventText(event) {
  return String(
    event?.text ?? event?.summary ?? event?.correction ?? event?.detail ?? event?.message ?? event?.value ?? "",
  ).trim();
}

function buildStarterQuestion(subject, language) {
  const topic = String(subject || "this topic").trim();
  const lang = String(language || "English").trim();
  return `Great. Let's practice ${topic} in ${lang}. First question: can you introduce your thoughts on ${topic} in 2-3 sentences?`;
}

function normalizeTtsStatus(payload) {
  const source = payload && typeof payload === "object" ? payload : {};
  const ready = Boolean(source.ready);
  const state = String(source.state || (ready ? "ready" : "idle")).trim().toLowerCase() || "idle";
  const detail = String(source.detail || "").trim() || (ready ? "Voice model ready." : "Voice model is preparing.");
  return {
    ok: Boolean(source.ok ?? true),
    engine: String(source.engine || "cosyvoice"),
    provider: String(source.provider || "cosyvoice"),
    ready,
    state,
    detail,
    modelId: String(source.model_id || ""),
    loadedModelId: String(source.loaded_model_id || ""),
    warmupActive: Boolean(source.warmup_active),
    updatedAt: source.updated_at ?? null,
  };
}

function normalizeRuntimeStatus(payload) {
  const source = payload && typeof payload === "object" ? payload : {};
  const components = source.components && typeof source.components === "object" ? source.components : {};
  const normalizeComponent = (name) => {
    const component = components[name] && typeof components[name] === "object" ? components[name] : {};
    return {
      ready: Boolean(component.ready),
      state: String(component.state || (component.ready ? "ready" : "idle")).toLowerCase(),
      detail: String(component.detail || "").trim(),
    };
  };
  return {
    ok: Boolean(source.ok ?? true),
    mode: String(source.mode || "idle"),
    ready: Boolean(source.ready),
    state: String(source.state || "idle"),
    components: {
      llm: normalizeComponent("llm"),
      asr: normalizeComponent("asr"),
      tts: normalizeComponent("tts"),
      languagetool: normalizeComponent("languagetool"),
    },
  };
}

function ttsStateLabel(state, ready) {
  if (ready) {
    return "Voice ready";
  }
  const normalized = String(state || "idle").trim().toLowerCase();
  if (normalized === "downloading") {
    return "Downloading voice model";
  }
  if (normalized === "loading") {
    return "Loading voice model";
  }
  if (normalized === "warming") {
    return "Warming up voice";
  }
  if (normalized === "error") {
    return "Voice service error";
  }
  return "Voice standby";
}

function mapStoredTurn(turn) {
  return {
    id: String(turn?.id || makeId("stored")),
    transcript: String(turn?.transcript || "").trim(),
    partialTranscript: "",
    reply: String(turn?.reply || "").trim(),
    question: "",
    correction: String(turn?.correction || "").trim(),
    feedback: String(turn?.explanation || "").trim(),
    mistakes: normalizeMistakes(turn?.mistakes),
    score: typeof turn?.score === "number" ? turn.score : null,
    error: "",
    startedAt: turn?.created_at || null,
    endedAt: turn?.created_at || null,
  };
}

function formatWhen(value) {
  const stamp = Date.parse(String(value || ""));
  if (Number.isNaN(stamp)) {
    return "";
  }
  return new Date(stamp).toLocaleString();
}

function getPersonaById(personaId) {
  return PERSONA_OPTIONS.find((item) => item.id === personaId) || PERSONA_OPTIONS[0];
}

function AuthGate() {
  return (
    <div className="coach-auth">
      <div className="coach-auth-card">
        <h2>Sign in first</h2>
        <p>
          This page uses <code>bipod_token</code>. Sign in from the main app, then return here.
        </p>
        <a className="coach-btn" href="/">
          Go to login
        </a>
      </div>
    </div>
  );
}

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { errorMessage: "" };
  }

  static getDerivedStateFromError(error) {
    return { errorMessage: String(error?.message || "Coach UI crashed.") };
  }

  componentDidCatch(error) {
    console.error("Coach UI crash:", error);
  }

  render() {
    if (this.state.errorMessage) {
      return (
        <div className="coach-auth">
          <div className="coach-auth-card">
            <h2>Coach UI failed</h2>
            <p>{this.state.errorMessage}</p>
            <button type="button" className="coach-btn" onClick={() => window.location.reload()}>
              Reload page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function CoachApp() {
  const [token, setToken] = useState(() => getInitialToken());
  const [authState, setAuthState] = useState(token ? "checking" : "missing");
  const [username, setUsername] = useState("Coach user");

  const [sessions, setSessions] = useState([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState("");

  const [subjectChoice, setSubjectChoice] = useState(SUBJECT_OPTIONS[0]);
  const [customSubject, setCustomSubject] = useState("");
  const [languageChoice, setLanguageChoice] = useState("English");
  const [supportedLanguages, setSupportedLanguages] = useState([]);
  const [personaChoice, setPersonaChoice] = useState(PERSONA_OPTIONS[0].id);
  const [voiceOptions, setVoiceOptions] = useState([]);
  const [voiceChoice, setVoiceChoice] = useState(DEFAULT_BUILTIN_VOICE_CHOICE);
  const [builtinVoices, setBuiltinVoices] = useState([]);
  const [voiceProfiles, setVoiceProfiles] = useState([]);
  const [referenceClipId, setReferenceClipId] = useState("");
  const [referenceClipTitle, setReferenceClipTitle] = useState("");
  const [profileDraftName, setProfileDraftName] = useState(DEFAULT_PROFILE_NAME);
  const [isUploadingReference, setIsUploadingReference] = useState(false);
  const [isCreatingProfile, setIsCreatingProfile] = useState(false);
  const [deletingProfileId, setDeletingProfileId] = useState("");
  const [autoSpeakReplies, setAutoSpeakReplies] = useState(true);

  const [sessionId, setSessionId] = useState("");
  const [activeSubject, setActiveSubject] = useState("");
  const [activeLanguage, setActiveLanguage] = useState("English");
  const [activePersonaId, setActivePersonaId] = useState(PERSONA_OPTIONS[0].id);
  const [activeSessionStatus, setActiveSessionStatus] = useState("active");
  const [starterQuestion, setStarterQuestion] = useState("");

  const [turns, setTurns] = useState([]);
  const [currentTurn, setCurrentTurn] = useState(null);
  const [textDraft, setTextDraft] = useState("");

  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isEnding, setIsEnding] = useState(false);

  const [statusMessage, setStatusMessage] = useState("Pick a subject and language, then start.");
  const [recordingDuration, setRecordingDuration] = useState("00:00");
  const [errorMessage, setErrorMessage] = useState("");
  const [summary, setSummary] = useState(null);
  const [conversationEnded, setConversationEnded] = useState(false);
  const [viewMode, setViewMode] = useState("chat");
  const [showSessionSettings, setShowSessionSettings] = useState(false);
  const [voiceStage, setVoiceStage] = useState({
    speaker: "ai",
    text: "",
    status: "idle",
  });
  const [ttsStatus, setTtsStatus] = useState(() => normalizeTtsStatus(null));
  const [runtimeStatus, setRuntimeStatus] = useState(() => normalizeRuntimeStatus(null));

  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const chunksRef = useRef([]);
  const recorderStartRef = useRef(0);
  const durationIntervalRef = useRef(null);
  const isStoppingRef = useRef(false);
  const currentTurnRef = useRef(null);
  const streamAbortRef = useRef(null);
  const threadRef = useRef(null);
  const activeLanguageRef = useRef("English");
  const autoSpeakRepliesRef = useRef(true);
  const audioPlaybackRef = useRef(null);
  const starterPlaybackKeyRef = useRef("");

  const selectedSubject = useMemo(() => {
    if (subjectChoice !== "Custom") {
      return subjectChoice;
    }
    return customSubject.trim();
  }, [subjectChoice, customSubject]);

  const selectedPersona = useMemo(() => getPersonaById(personaChoice), [personaChoice]);
  const activePersona = useMemo(() => getPersonaById(activePersonaId), [activePersonaId]);
  const selectedBuiltinVoice = useMemo(() => {
    const parsed = parseVoiceChoice(voiceChoice);
    if (!parsed.builtinVoiceId) {
      return null;
    }
    return (
      builtinVoices.find((voice) => String(voice.id || "").toLowerCase() === String(parsed.builtinVoiceId).toLowerCase())
      || null
    );
  }, [voiceChoice, builtinVoices]);

  const voiceStageSnapshot = useMemo(() => {
    const stageText = String(voiceStage.text || "").trim();
    if (stageText) {
      return {
        speaker: voiceStage.speaker || "ai",
        text: stageText,
        status: voiceStage.status || "idle",
      };
    }

    const userText = String(currentTurn?.transcript || currentTurn?.partialTranscript || "").trim();
    if (userText) {
      return {
        speaker: "user",
        text: userText,
        status: isRecording ? "listening" : "processing",
      };
    }

    const latestTurn = turns.length ? turns[turns.length - 1] : null;
    const latestReply = String(latestTurn?.reply || "").trim();
    if (latestReply) {
      return {
        speaker: "ai",
        text: latestReply,
        status: "idle",
      };
    }

    return {
      speaker: "ai",
      text: String(starterQuestion || "Waiting for the conversation to begin."),
      status: "idle",
    };
  }, [voiceStage, currentTurn, isRecording, turns, starterQuestion]);

  useEffect(() => {
    if (!token) {
      setAuthState("missing");
      return;
    }

    let cancelled = false;

    async function fetchIdentity() {
      setAuthState("checking");
      try {
        const response = await fetch("/api/v1/auth/me", { headers: buildAuthHeaders(token) });
        if (!response.ok) {
          if (!cancelled) {
            if (response.status === 401) {
              setAuthState("invalid");
              setToken("");
            } else {
              setAuthState("ready");
            }
          }
          return;
        }
        const payload = await response.json();
        if (!cancelled) {
          setUsername(payload.username || "Coach user");
          setAuthState("ready");
        }
      } catch {
        if (!cancelled) {
          setAuthState("ready");
        }
      }
    }

    fetchIdentity();

    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    if (authState !== "ready" || !token) {
      return;
    }
    refreshSessions();
    refreshSupportedLanguages();
    refreshBuiltinVoices();
    refreshVoiceProfiles();
    void preloadCoachRuntime("text", { silent: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authState, token]);

  useEffect(() => {
    if (authState !== "ready" || !token) {
      setTtsStatus(normalizeTtsStatus(null));
      return;
    }

    let cancelled = false;
    const poll = async () => {
      if (cancelled) {
        return;
      }
      const shouldWarmTts = Boolean(sessionId) && viewMode === "voice";
      await refreshTtsStatus({ warm: shouldWarmTts, silent: true });
      const runtimeMode = sessionId ? (viewMode === "voice" ? "voice" : "text") : "idle";
      await refreshRuntimeStatus({ warm: false, mode: runtimeMode, silent: true });
    };
    void poll();
    const intervalId = window.setInterval(() => {
      void poll();
    }, 12000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authState, token, sessionId, viewMode]);

  useEffect(() => {
    if (!sessionId || !token || authState !== "ready") {
      return;
    }
    const mode = viewMode === "voice" ? "voice" : "text";
    void preloadCoachRuntime(mode, { silent: true });
  }, [sessionId, viewMode, token, authState]);

  useEffect(() => {
    activeLanguageRef.current = activeLanguage || languageChoice || "English";
  }, [activeLanguage, languageChoice]);

  useEffect(() => {
    autoSpeakRepliesRef.current = Boolean(autoSpeakReplies);
  }, [autoSpeakReplies]);

  useEffect(() => {
    const targetPersonaId = VOICE_TO_PERSONA_ID[String(voiceChoice || "")];
    if (!targetPersonaId) {
      return;
    }
    if (!PERSONA_OPTIONS.some((item) => item.id === targetPersonaId)) {
      return;
    }
    setPersonaChoice((current) => (current === targetPersonaId ? current : targetPersonaId));
    setActivePersonaId((current) => (current === targetPersonaId ? current : targetPersonaId));
  }, [voiceChoice]);

  useEffect(() => {
    if (!sessionId || !starterQuestion || conversationEnded) {
      return;
    }
    const key = `${sessionId}:${starterQuestion}`;
    if (starterPlaybackKeyRef.current === key) {
      return;
    }
    starterPlaybackKeyRef.current = key;
    setVoiceStage({
      speaker: "ai",
      text: starterQuestion,
      status: "speaking",
    });
    void speakCoachText(starterQuestion, { force: true, source: "starter" });
  }, [sessionId, starterQuestion, conversationEnded]);

  useEffect(() => {
    if (!window.speechSynthesis) {
      return;
    }

    const synth = window.speechSynthesis;

    const loadVoices = () => {
      const voices = synth.getVoices() || [];
      const mapped = voices
        .filter((voice) => voice?.name)
        .map((voice) => ({
          name: voice.name,
          lang: voice.lang || "",
          localService: Boolean(voice.localService),
        }));
      setVoiceOptions(mapped);
      if (!mapped.length) {
        setVoiceChoice((current) => {
          if (
            current
            && (
              current.startsWith("server:")
              || current.startsWith("profile:")
              || current.startsWith("builtin:")
              || current === "session_clone"
            )
          ) {
            return current;
          }
          return DEFAULT_SERVER_VOICE;
        });
        return;
      }
      setVoiceChoice((current) => {
        if (
          current
          && (
            current.startsWith("server:")
            || current.startsWith("profile:")
            || current.startsWith("builtin:")
            || current === "session_clone"
          )
        ) {
          return current;
        }
        if (current && mapped.some((voice) => voice.name === current)) {
          return current;
        }
        return mapped[0].name;
      });
    };

    loadVoices();
    if ("onvoiceschanged" in synth) {
      synth.onvoiceschanged = loadVoices;
    }

    return () => {
      if ("onvoiceschanged" in synth) {
        synth.onvoiceschanged = null;
      }
      synth.cancel();
    };
  }, []);

  useEffect(() => {
    return () => {
      stopRecording(true);
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
      if (audioPlaybackRef.current) {
        audioPlaybackRef.current.pause();
        audioPlaybackRef.current = null;
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function keyDown(event) {
      if (!sessionId || conversationEnded || isSending || isEnding) {
        return;
      }
      if (event.code !== "Space" || event.repeat) {
        return;
      }
      const targetTag = String(event.target?.tagName || "").toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") {
        return;
      }
      event.preventDefault();
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    }

    window.addEventListener("keydown", keyDown);
    return () => {
      window.removeEventListener("keydown", keyDown);
    };
  }, [sessionId, conversationEnded, isSending, isEnding, isRecording]);

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [turns, currentTurn, starterQuestion]);

  async function refreshSessions() {
    if (!token) {
      return;
    }
    setIsLoadingSessions(true);
    try {
      const payload = await apiFetchJson("/api/v1/coach/sessions", token);
      if (Array.isArray(payload)) {
        setSessions(payload);
      }
    } catch {
      // Keep current UI state if session list refresh fails.
    } finally {
      setIsLoadingSessions(false);
    }
  }

  async function refreshSupportedLanguages() {
    if (!token) {
      return;
    }
    try {
      const payload = await apiFetchJson("/api/v1/coach/languages/supported", token);
      if (!Array.isArray(payload) || !payload.length) {
        return;
      }
      const selectable = payload.filter((item) => Boolean(item?.selectable) && item?.name);
      setSupportedLanguages(selectable);
      setLanguageChoice((current) => {
        const found = selectable.find((item) => String(item.name) === String(current));
        if (found) {
          return current;
        }
        const defaultItem = selectable.find((item) => Boolean(item?.is_default));
        return String(defaultItem?.name || selectable[0]?.name || current || "English");
      });
    } catch {
      // Keep static fallback list.
    }
  }

  async function refreshBuiltinVoices() {
    if (!token) {
      return;
    }
    try {
      const payload = await apiFetchJson("/api/v1/coach/voices/library", token);
      if (!Array.isArray(payload)) {
        return;
      }
      const available = payload.filter((item) => Boolean(item?.is_available));
      setBuiltinVoices(available);
      const anby = available.find((item) => String(item?.id || "").toLowerCase() === "anby");
      setVoiceChoice((current) => {
        const currentChoice = String(current || "");
        const builtinChoices = new Set(available.map((item) => String(item.choice_id || `builtin:${item.id}`)));
        if (builtinChoices.has(currentChoice)) {
          return currentChoice;
        }
        if (currentChoice.startsWith("server:") || currentChoice === "session_clone" || currentChoice.startsWith("profile:")) {
          if (currentChoice !== DEFAULT_SERVER_VOICE) {
            return currentChoice;
          }
          return anby?.choice_id || currentChoice;
        }
        if (currentChoice && voiceOptions.some((voice) => voice.name === currentChoice)) {
          return currentChoice;
        }
        return anby?.choice_id || DEFAULT_SERVER_VOICE;
      });
    } catch {
      // keep existing state if built-in library fails
    }
  }

  async function refreshVoiceProfiles() {
    if (!token) {
      return;
    }
    try {
      const payload = await apiFetchJson("/api/v1/coach/voices/profiles", token);
      if (Array.isArray(payload)) {
        setVoiceProfiles(payload);
      }
    } catch {
      // Keep current list if fetch fails.
    }
  }

  async function refreshTtsStatus({ warm = false, silent = true } = {}) {
    if (!token) {
      return null;
    }
    const path = warm ? "/api/v1/coach/tts/status?warm=true" : "/api/v1/coach/tts/status";
    try {
      const payload = await apiFetchJson(path, token);
      const normalized = normalizeTtsStatus(payload);
      setTtsStatus(normalized);
      return normalized;
    } catch (error) {
      const fallback = normalizeTtsStatus({
        ok: false,
        engine: "cosyvoice",
        provider: "cosyvoice",
        ready: false,
        state: "error",
        detail: error?.message || "Voice status is unavailable.",
      });
      setTtsStatus(fallback);
      if (!silent) {
        setErrorMessage(fallback.detail);
      }
      return fallback;
    }
  }

  async function refreshRuntimeStatus({ warm = false, mode = "voice", silent = true } = {}) {
    if (!token) {
      return null;
    }
    const query = new URLSearchParams();
    if (warm) {
      query.set("warm", "true");
    }
    if (mode) {
      query.set("mode", mode);
    }
    const path = `/api/v1/coach/runtime/status${query.toString() ? `?${query.toString()}` : ""}`;
    try {
      const payload = await apiFetchJson(path, token);
      const normalized = normalizeRuntimeStatus(payload);
      setRuntimeStatus(normalized);
      return normalized;
    } catch (error) {
      if (!silent) {
        setErrorMessage(error?.message || "Runtime status is unavailable.");
      }
      return null;
    }
  }

  async function preloadCoachRuntime(mode = "voice", { silent = true } = {}) {
    if (!token) {
      return null;
    }
    try {
      const payload = await apiFetchJson("/api/v1/coach/runtime/preload", token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      setRuntimeStatus(normalizeRuntimeStatus(payload));
      return payload;
    } catch (error) {
      if (!silent) {
        setErrorMessage(error?.message || "Runtime preload failed.");
      }
      return null;
    }
  }

  function hydrateSession(record, loadedTurns = []) {
    const subject = String(record?.focus_area || record?.title || "Conversation").trim() || "Conversation";
    const language = String(record?.target_language || "English").trim() || "English";
    const status = String(record?.status || "active").trim() || "active";

    setSessionId(String(record?.id || ""));
    setActiveSubject(subject);
    setActiveLanguage(language);
    setActivePersonaId(personaChoice);
    setActiveSessionStatus(status);
    const linkedProfileId = String(record?.voice_profile_id || "").trim();
    if (linkedProfileId) {
      if (linkedProfileId.startsWith("builtin:")) {
        setVoiceChoice(linkedProfileId);
      } else {
        setVoiceChoice(`profile:${linkedProfileId}`);
      }
    }
    setConversationEnded(status === "completed");
    setShowSessionSettings(false);
    setStarterQuestion(buildStarterQuestion(subject, language));
    starterPlaybackKeyRef.current = "";
    setSummary(null);
    setCurrentTurn(null);
    currentTurnRef.current = null;
    setTurns(Array.isArray(loadedTurns) ? loadedTurns : []);
    setVoiceStage({
      speaker: "ai",
      text: buildStarterQuestion(subject, language),
      status: "idle",
    });
    setStatusMessage(
      status === "completed"
        ? "This session is completed."
        : `Session ready in ${language}. Click the mic to start and click again to stop.`,
    );
  }

  async function openSession(record) {
    if (!record?.id || !token) {
      return;
    }

    setErrorMessage("");
    setIsSending(false);
    stopRecording(true);

    try {
      const rawTurns = await apiFetchJson(`/api/v1/coach/sessions/${record.id}/turns`, token);
      const mappedTurns = Array.isArray(rawTurns) ? rawTurns.map(mapStoredTurn) : [];
      hydrateSession(record, mappedTurns);
    } catch (error) {
      setErrorMessage(error?.message || "Could not load this session.");
    }
  }

  async function deleteSession(record, event) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const id = String(record?.id || "");
    if (!id || !token || deletingSessionId) {
      return;
    }
    setErrorMessage("");
    setDeletingSessionId(id);
    try {
      await apiFetchJson(`/api/v1/coach/sessions/${id}`, token, { method: "DELETE" });
      setSessions((prev) => prev.filter((item) => item.id !== id));
      if (id === sessionId) {
        startAnotherConversation();
      }
    } catch (error) {
      setErrorMessage(error?.message || "Could not delete this session.");
    } finally {
      setDeletingSessionId("");
    }
  }

  async function uploadReferenceClip(file) {
    if (!file || !token || isUploadingReference) {
      return;
    }
    setErrorMessage("");
    setIsUploadingReference(true);
    try {
      const formData = new FormData();
      formData.append("file", file, file.name || "voice-sample.wav");
      formData.append("title", file.name || "Voice sample");
      formData.append("language", activeLanguage || languageChoice || "English");
      const response = await fetch("/api/v1/coach/voices/reference", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(text || `Upload failed (${response.status})`);
      }
      const payload = await response.json();
      setReferenceClipId(String(payload?.id || ""));
      setReferenceClipTitle(String(payload?.title || file.name || "Voice sample"));
      setVoiceChoice("session_clone");
      setStatusMessage("Voice sample uploaded. Save it as a profile or use it for this session.");
    } catch (error) {
      setErrorMessage(error?.message || "Could not upload voice sample.");
    } finally {
      setIsUploadingReference(false);
    }
  }

  async function createVoiceProfileFromReference() {
    if (!referenceClipId || !token || isCreatingProfile) {
      return;
    }
    const trimmedName = String(profileDraftName || "").trim();
    if (!trimmedName) {
      setErrorMessage("Profile name is required.");
      return;
    }
    setErrorMessage("");
    setIsCreatingProfile(true);
    try {
      const payload = await apiFetchJson("/api/v1/coach/voices/profiles", token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trimmedName,
          reference_clip_id: referenceClipId,
          language: activeLanguage || languageChoice || "English",
        }),
      });
      setVoiceProfiles((prev) => [payload, ...prev.filter((item) => item.id !== payload.id)]);
      setVoiceChoice(`profile:${payload.id}`);
      setProfileDraftName(DEFAULT_PROFILE_NAME);
      setStatusMessage(`Voice profile "${payload.name}" is ready.`);
    } catch (error) {
      setErrorMessage(error?.message || "Could not create voice profile.");
    } finally {
      setIsCreatingProfile(false);
    }
  }

  async function deleteVoiceProfile(profileId) {
    const id = String(profileId || "");
    if (!id || !token || deletingProfileId) {
      return;
    }
    setErrorMessage("");
    setDeletingProfileId(id);
    try {
      await apiFetchJson(`/api/v1/coach/voices/profiles/${id}`, token, { method: "DELETE" });
      setVoiceProfiles((prev) => prev.filter((item) => item.id !== id));
      if (voiceChoice === `profile:${id}`) {
        setVoiceChoice(DEFAULT_SERVER_VOICE);
      }
    } catch (error) {
      setErrorMessage(error?.message || "Could not delete voice profile.");
    } finally {
      setDeletingProfileId("");
    }
  }

  function stopCurrentPlayback() {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (audioPlaybackRef.current) {
      audioPlaybackRef.current.pause();
      audioPlaybackRef.current = null;
    }
  }

  async function playServerTts(text) {
    if (!token) {
      return false;
    }
    const content = String(text || "").trim();
    if (!content) {
      return false;
    }

    const parsedVoice = parseVoiceChoice(voiceChoice);
    const voicePreset = parsedVoice.voicePreset || "default";
    const language = activeLanguageRef.current || activeLanguage || languageChoice || "English";
    const persona = activePersona?.prompt || selectedPersona?.prompt || "";
    if (parsedVoice.voiceMode === "cloned_session" && !referenceClipId) {
      throw new Error("Upload a reference voice sample first.");
    }
    const payload = {
      text: content,
      language,
      voice_preset: voicePreset,
      persona_style: persona,
      tts_provider: "cosyvoice",
      voice_mode: parsedVoice.voiceMode,
      voice_profile_id: parsedVoice.voiceProfileId || null,
      reference_clip_id: parsedVoice.voiceMode === "cloned_session" ? referenceClipId || null : null,
      builtin_voice_id: parsedVoice.builtinVoiceId || null,
    };

    const response = await fetch("/api/v1/coach/tts", {
      method: "POST",
      headers: buildAuthHeaders(token, { "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      void refreshTtsStatus({ warm: true, silent: true });
      const raw = await response.text().catch(() => "");
      let detail = raw || `TTS failed (${response.status})`;
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object" && typeof parsed.detail === "string" && parsed.detail.trim()) {
            detail = parsed.detail.trim();
          }
        } catch {
          // keep raw response
        }
      }
      throw new Error(detail);
    }

    const audioBlob = await response.blob();
    if (!audioBlob.size) {
      throw new Error("TTS returned empty audio");
    }

    stopCurrentPlayback();
    setVoiceStage({
      speaker: "ai",
      text: content,
      status: "speaking",
    });
    const audioUrl = window.URL.createObjectURL(audioBlob);
    const player = new Audio(audioUrl);
    audioPlaybackRef.current = player;
    player.onended = () => {
      if (audioPlaybackRef.current === player) {
        audioPlaybackRef.current = null;
      }
      setVoiceStage((current) => ({ ...current, status: "idle" }));
      window.URL.revokeObjectURL(audioUrl);
    };
    player.onerror = () => {
      if (audioPlaybackRef.current === player) {
        audioPlaybackRef.current = null;
      }
      setVoiceStage((current) => ({ ...current, status: "idle" }));
      window.URL.revokeObjectURL(audioUrl);
    };
    await player.play();
    return true;
  }

  async function speakCoachText(text, options = {}) {
    const content = String(text || "").trim();
    const forcePlay = Boolean(options?.force);
    if (!content || (!forcePlay && !autoSpeakRepliesRef.current)) {
      return;
    }
    setVoiceStage({
      speaker: "ai",
      text: content,
      status: "speaking",
    });

    const parsedVoice = parseVoiceChoice(voiceChoice);
    const shouldUseServer = !parsedVoice.useBrowserVoice || parsedVoice.voiceMode !== "preset";
    if (!shouldUseServer && window.speechSynthesis && window.SpeechSynthesisUtterance) {
      try {
        const utterance = new window.SpeechSynthesisUtterance(content);
        const activeLang = activeLanguageRef.current || "English";
        utterance.lang = LANGUAGE_TO_BCP47[activeLang] || "en-US";

        const voices = window.speechSynthesis.getVoices() || [];
        const selectedVoice = voices.find((voice) => voice.name === voiceChoice);
        if (selectedVoice) {
          utterance.voice = selectedVoice;
        }
        if (selectedVoice || voices.length) {
          stopCurrentPlayback();
          utterance.onend = () => {
            setVoiceStage((current) => ({ ...current, status: "idle" }));
          };
          utterance.onerror = () => {
            setVoiceStage((current) => ({ ...current, status: "idle" }));
          };
          window.speechSynthesis.speak(utterance);
          return;
        }
      } catch {
        // Fallback to server TTS below.
      }
    }

    try {
      await playServerTts(content);
    } catch (error) {
      void refreshTtsStatus({ warm: true, silent: true });
      setVoiceStage((current) => ({ ...current, status: "idle" }));
      if (forcePlay) {
        setErrorMessage(error?.message || "Could not play AI voice.");
      }
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
    } catch {
      releaseStream();
      setIsRecording(false);
    }
  }

  async function startConversation() {
    if (!selectedSubject) {
      setErrorMessage("Pick a subject first.");
      return;
    }
    if (!languageChoice) {
      setErrorMessage("Pick a language first.");
      return;
    }
    if (!token) {
      setAuthState("missing");
      return;
    }

    setErrorMessage("");
    setSummary(null);
    const parsedVoice = parseVoiceChoice(voiceChoice);
    const linkedProfileId = parsedVoice.voiceMode === "cloned_profile"
      ? parsedVoice.voiceProfileId
      : (parsedVoice.builtinVoiceId ? `builtin:${parsedVoice.builtinVoiceId}` : null);

    try {
      const created = await apiFetchJson("/api/v1/coach/sessions", token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: `${selectedSubject} conversation`,
          target_language: languageChoice,
          cefr_level: "B1",
          focus_area: selectedSubject,
          voice_profile_id: linkedProfileId,
        }),
      });

      const sessionRecord = {
        ...created,
        focus_area: created?.focus_area || selectedSubject,
        target_language: created?.target_language || languageChoice,
      };

      setSessions((prev) => [sessionRecord, ...prev.filter((item) => item.id !== sessionRecord.id)]);
      hydrateSession(sessionRecord, []);
      void preloadCoachRuntime("voice", { silent: true });
      setStatusMessage(`Conversation started on ${selectedSubject} in ${languageChoice}. Click the mic to start and click again to stop.`);
    } catch (error) {
      setErrorMessage(error?.message || "Could not start conversation.");
    }
  }

  function beginDraftTurn() {
    const draft = {
      id: makeId("turn"),
      transcript: "",
      partialTranscript: "",
      reply: "",
      question: "",
      correction: "",
      feedback: "",
      mistakes: [],
      score: null,
      error: "",
      startedAt: new Date().toISOString(),
      endedAt: null,
    };
    currentTurnRef.current = draft;
    setCurrentTurn({ ...draft });
  }

  function updateDraft(mutator) {
    const base = currentTurnRef.current;
    if (!base) {
      return;
    }
    const next = mutator({ ...base });
    currentTurnRef.current = next;
    setCurrentTurn({ ...next });
  }

  async function startRecording() {
    if (!sessionId || conversationEnded || isRecording || isSending || isEnding) {
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setErrorMessage("This browser does not support audio capture.");
      return;
    }

    setErrorMessage("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
      const mimeType = mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || "";
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;
      mediaStreamRef.current = stream;
      chunksRef.current = [];
      recorderStartRef.current = Date.now();
      isStoppingRef.current = false;

      beginDraftTurn();

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setErrorMessage("Recorder error. Please try again.");
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        releaseStream();
        setIsRecording(false);
        if (!isStoppingRef.current) {
          await sendAudioTurn(audioBlob);
        }
      };

      recorder.start(250);
      setIsRecording(true);
      setRecordingDuration("00:00");
      setStatusMessage("I'm listening...");
      setVoiceStage({
        speaker: "user",
        text: "I'm listening...",
        status: "listening",
      });
      durationIntervalRef.current = window.setInterval(() => {
        const elapsed = Date.now() - recorderStartRef.current;
        const totalSeconds = Math.max(0, Math.floor(elapsed / 1000));
        const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
        const seconds = String(totalSeconds % 60).padStart(2, "0");
        setRecordingDuration(`${minutes}:${seconds}`);
      }, 250);
    } catch (error) {
      releaseStream();
      setIsRecording(false);
      setErrorMessage(error?.message || "Microphone access failed.");
    }
  }

  async function sendAudioTurn(audioBlob) {
    if (!sessionId || !token || !currentTurnRef.current) {
      return;
    }

    setIsSending(true);
    setStatusMessage("AI is reviewing your answer...");

    const controller = new AbortController();
    streamAbortRef.current = controller;

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, `coach-${currentTurnRef.current.id}.webm`);
      formData.append("file", audioBlob, `coach-${currentTurnRef.current.id}.webm`);
      formData.append("session_id", sessionId);
      formData.append("persona_style", activePersona.prompt);

      const response = await fetch("/api/v1/coach/turns/stream", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
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
          } catch {
            continue;
          }

          const eventType = String(event?.type || "").toLowerCase();
          if (eventType === "stt_partial") {
            const partialText = eventText(event);
            updateDraft((draft) => {
              draft.partialTranscript = partialText;
              return draft;
            });
            if (partialText) {
              setVoiceStage({
                speaker: "user",
                text: partialText,
                status: "listening",
              });
            }
            continue;
          }

          if (eventType === "stt_final") {
            const finalText = eventText(event);
            const confidenceBand = String(event?.asr_confidence_band || "").toLowerCase();
            updateDraft((draft) => {
              draft.transcript = finalText || draft.transcript;
              draft.partialTranscript = "";
              if (confidenceBand) {
                draft.asrConfidenceBand = confidenceBand;
              }
              return draft;
            });
            if (finalText) {
              setVoiceStage({
                speaker: "user",
                text: finalText,
                status: "processing",
              });
            }
            if (confidenceBand === "low") {
              setStatusMessage("ASR is uncertain. The coach may ask for confirmation.");
            }
            continue;
          }

          if (eventType === "coach_reply") {
            const replyText = eventText(event);
            updateDraft((draft) => {
              draft.reply = replyText || draft.reply;
              draft.question = String(event?.question || "").trim();
              return draft;
            });
            if (replyText) {
              setVoiceStage({
                speaker: "ai",
                text: replyText,
                status: "speaking",
              });
              void speakCoachText(replyText);
            }
            continue;
          }

          if (eventType === "feedback") {
            updateDraft((draft) => {
              draft.feedback = String(event?.summary || "").trim();
              draft.correction = String(event?.correction || "").trim();
              draft.mistakes = normalizeMistakes(event?.mistakes);
              return draft;
            });
            continue;
          }

          if (eventType === "score") {
            updateDraft((draft) => {
              const rawValue = event?.value;
              if (rawValue == null || rawValue === "") {
                draft.score = null;
                return draft;
              }
              const valueRaw = Number(rawValue);
              draft.score = Number.isNaN(valueRaw) ? null : Math.max(0, Math.min(100, valueRaw));
              return draft;
            });
            continue;
          }

          if (eventType === "error") {
            throw new Error(eventText(event) || "Coach stream failed.");
          }

          if (eventType === "done") {
            streamComplete = true;
            break;
          }
        }
      }

      const finalTurn = {
        ...currentTurnRef.current,
        transcript: currentTurnRef.current?.transcript || currentTurnRef.current?.partialTranscript || "",
        endedAt: new Date().toISOString(),
      };
      setTurns((prev) => [...prev, finalTurn]);
      setCurrentTurn(null);
      currentTurnRef.current = null;

      if (finalTurn.score == null) {
        setStatusMessage("No clear speech detected. Please answer again.");
        setVoiceStage({
          speaker: "ai",
          text: "I couldn't catch clear speech. Please answer again with one full sentence.",
          status: "idle",
        });
      } else {
        setStatusMessage("Your turn. Answer the next question.");
        if (finalTurn.reply) {
          setVoiceStage((current) => ({
            speaker: "ai",
            text: finalTurn.reply,
            status: current.status === "speaking" ? current.status : "idle",
          }));
        }
      }
    } catch (error) {
      updateDraft((draft) => {
        draft.error = error?.message || "Failed to process this answer.";
        return draft;
      });
      const failedTurn = {
        ...currentTurnRef.current,
        endedAt: new Date().toISOString(),
      };
      setTurns((prev) => [...prev, failedTurn]);
      setCurrentTurn(null);
      currentTurnRef.current = null;
      setErrorMessage(error?.message || "Failed to stream this turn.");
      setStatusMessage("Turn failed. Try again.");
      setVoiceStage({
        speaker: "ai",
        text: error?.message || "Turn failed. Please try again.",
        status: "idle",
      });
    } finally {
      setIsSending(false);
      streamAbortRef.current = null;
      refreshSessions();
    }
  }

  async function sendTextTurn() {
    if (!sessionId || !token || isSending || isEnding || conversationEnded) {
      return;
    }
    const message = String(textDraft || "").trim();
    if (!message) {
      return;
    }
    setErrorMessage("");
    setIsSending(true);
    setStatusMessage("AI is reviewing your text...");

    const draft = {
      id: makeId("text"),
      transcript: message,
      partialTranscript: "",
      reply: "",
      question: "",
      correction: "",
      feedback: "",
      mistakes: [],
      score: null,
      error: "",
      startedAt: new Date().toISOString(),
      endedAt: null,
    };
    currentTurnRef.current = draft;
    setCurrentTurn({ ...draft });

    try {
      const payload = await apiFetchJson("/api/v1/coach/turns/text", token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          text: message,
          persona_style: activePersona.prompt,
        }),
      });

      const mappedTurn = mapStoredTurn(payload);
      mappedTurn.id = String(payload?.id || draft.id);
      mappedTurn.transcript = message;
      mappedTurn.endedAt = new Date().toISOString();
      setTurns((prev) => [...prev, mappedTurn]);
      setCurrentTurn(null);
      currentTurnRef.current = null;
      setTextDraft("");
      setStatusMessage("Your turn. Answer the next question.");
      if (mappedTurn.reply) {
        setVoiceStage({
          speaker: "ai",
          text: mappedTurn.reply,
          status: "speaking",
        });
        void speakCoachText(mappedTurn.reply);
      }
    } catch (error) {
      updateDraft((next) => {
        next.error = error?.message || "Failed to process this text turn.";
        return next;
      });
      const failedTurn = {
        ...currentTurnRef.current,
        endedAt: new Date().toISOString(),
      };
      setTurns((prev) => [...prev, failedTurn]);
      setCurrentTurn(null);
      currentTurnRef.current = null;
      setErrorMessage(error?.message || "Text turn failed.");
      setStatusMessage("Turn failed. Try again.");
      setVoiceStage({
        speaker: "ai",
        text: error?.message || "Turn failed. Please try again.",
        status: "idle",
      });
    } finally {
      setIsSending(false);
      refreshSessions();
    }
  }

  function handleMicToggle(event) {
    event.preventDefault();
    if (isRecording) {
      setVoiceStage((current) => ({
        speaker: "user",
        text: current.text || "Processing your answer...",
        status: "processing",
      }));
      stopRecording();
    } else {
      startRecording();
    }
  }

  async function endConversation() {
    if (!sessionId || !token || isSending || isRecording) {
      return;
    }

    setIsEnding(true);
    setErrorMessage("");

    try {
      const result = await apiFetchJson(`/api/v1/coach/sessions/${sessionId}/end`, token, { method: "POST" });
      setSummary(result);
      setConversationEnded(true);
      setActiveSessionStatus("completed");
      setStatusMessage("Conversation ended. Review your score and feedback.");
      setSessions((prev) => prev.map((item) => (item.id === sessionId ? { ...item, status: "completed" } : item)));
    } catch (error) {
      setErrorMessage(error?.message || "Could not end conversation.");
    } finally {
      setIsEnding(false);
      refreshSessions();
    }
  }

  function startAnotherConversation() {
    stopRecording(true);
    starterPlaybackKeyRef.current = "";
    setSessionId("");
    setActiveSubject("");
    setActiveLanguage(languageChoice || "English");
    setActivePersonaId(personaChoice);
    setActiveSessionStatus("active");
    setStarterQuestion("");
    setTurns([]);
    setCurrentTurn(null);
    currentTurnRef.current = null;
    setTextDraft("");
    setSummary(null);
    setConversationEnded(false);
    setShowSessionSettings(false);
    setViewMode("chat");
    setReferenceClipId("");
    setReferenceClipTitle("");
    setProfileDraftName(DEFAULT_PROFILE_NAME);
    setVoiceChoice((current) => {
      const builtinIds = new Set(builtinVoices.map((voice) => String(voice.choice_id || `builtin:${voice.id}`)));
      if (builtinIds.has(DEFAULT_BUILTIN_VOICE_CHOICE)) {
        return DEFAULT_BUILTIN_VOICE_CHOICE;
      }
      if (builtinIds.has(current)) {
        return current;
      }
      return DEFAULT_SERVER_VOICE;
    });
    setStatusMessage("Pick a subject and language, then start.");
    setVoiceStage({
      speaker: "ai",
      text: "Choose a subject and start to hear the AI coach.",
      status: "idle",
    });
    setErrorMessage("");
  }

  if (authState === "missing" || authState === "invalid") {
    return <AuthGate />;
  }

  const ttsBadgeClass = `coach-tts-badge ${ttsStatus.ready ? "ready" : ""} state-${ttsStatus.state || "idle"}`;
  const aiSpeakingNow = voiceStageSnapshot.speaker === "ai" && voiceStageSnapshot.status === "speaking";

  return (
    <div className="coach-shell">
      <aside className="coach-sidebar">
        <div className="coach-sidebar-top">
          <h2>Sessions</h2>
          <button type="button" className="coach-btn coach-btn-small" onClick={startAnotherConversation}>
            New session
          </button>
        </div>
        {isLoadingSessions ? <p className="coach-side-note">Loading history...</p> : null}
        <div className="coach-session-list">
          {sessions.length ? (
            sessions.map((item) => {
              const isActive = item.id === sessionId;
              const when = formatWhen(item.updated_at || item.created_at);
              const isDeleting = deletingSessionId === item.id;
              return (
                <div key={item.id} className={`coach-session-item ${isActive ? "active" : ""}`}>
                  <button type="button" className="coach-session-open" onClick={() => openSession(item)}>
                    <div className="coach-session-title">{item.focus_area || item.title || "Untitled session"}</div>
                    <div className="coach-session-meta">
                      <span>{item.target_language || "English"}</span>
                      <span>{item.turn_count || 0} turns</span>
                      <span>{String(item.status || "active")}</span>
                    </div>
                    {when ? <div className="coach-session-time">{when}</div> : null}
                  </button>
                  <div className="coach-session-actions">
                    <button
                      type="button"
                      className="coach-delete-btn"
                      onClick={(event) => deleteSession(item, event)}
                      aria-label="Delete session"
                      title="Delete session"
                      disabled={isDeleting}
                    >
                      <span className="material-symbols-rounded">{isDeleting ? "hourglass_top" : "delete"}</span>
                    </button>
                  </div>
                </div>
              );
            })
          ) : (
            <p className="coach-side-note">No sessions yet.</p>
          )}
        </div>
      </aside>

      <main className="coach-main">
        <div className="coach-top">
          <button type="button" className="coach-back-btn" onClick={() => (window.location.href = "/")}>
            <span className="material-symbols-rounded">arrow_back</span>
            Back to chat
          </button>
          <div className="coach-top-head">
            <h1>AI Speaking Coach</h1>
            <div className={ttsBadgeClass} title={ttsStatus.detail}>
              <span className="coach-tts-badge-dot" />
              <div className="coach-tts-badge-text">
                <strong>{ttsStateLabel(ttsStatus.state, ttsStatus.ready)}</strong>
                <span>{ttsStatus.detail}</span>
              </div>
            </div>
          </div>
          <p>{statusMessage}</p>
          <div className="coach-runtime-row" aria-label="Runtime readiness">
            <span className={`coach-runtime-chip ${runtimeStatus.components.llm.ready ? "ready" : ""}`}>
              LLM {runtimeStatus.components.llm.ready ? "ready" : runtimeStatus.components.llm.state}
            </span>
            <span className={`coach-runtime-chip ${runtimeStatus.components.asr.ready ? "ready" : ""}`}>
              ASR {runtimeStatus.components.asr.ready ? "ready" : runtimeStatus.components.asr.state}
            </span>
            <span className={`coach-runtime-chip ${runtimeStatus.components.tts.ready ? "ready" : ""}`}>
              TTS {runtimeStatus.components.tts.ready ? "ready" : runtimeStatus.components.tts.state}
            </span>
            <span className={`coach-runtime-chip ${runtimeStatus.components.languagetool.ready ? "ready" : ""}`}>
              LT {runtimeStatus.components.languagetool.ready ? "ready" : runtimeStatus.components.languagetool.state}
            </span>
          </div>
        </div>

        {!sessionId ? (
          <section className="coach-setup-card">
            <label htmlFor="subject-select">Subject</label>
            <select id="subject-select" value={subjectChoice} onChange={(event) => setSubjectChoice(event.target.value)}>
              {SUBJECT_OPTIONS.map((subject) => (
                <option key={subject} value={subject}>
                  {subject}
                </option>
              ))}
            </select>

            {subjectChoice === "Custom" ? (
              <input
                type="text"
                placeholder="Type your subject"
                value={customSubject}
                onChange={(event) => setCustomSubject(event.target.value)}
                maxLength={120}
              />
            ) : null}

            <label htmlFor="language-select">Language (supported)</label>
            <select id="language-select" value={languageChoice} onChange={(event) => setLanguageChoice(event.target.value)}>
              {(supportedLanguages.length ? supportedLanguages.map((item) => String(item.name || "")) : LANGUAGE_OPTIONS).map((language) => (
                <option key={language} value={language}>
                  {language}
                </option>
              ))}
            </select>

            <label htmlFor="persona-select">AI personality</label>
            <select id="persona-select" value={personaChoice} onChange={(event) => setPersonaChoice(event.target.value)}>
              {PERSONA_OPTIONS.map((persona) => (
                <option key={persona.id} value={persona.id}>
                  {persona.label}
                </option>
              ))}
            </select>

            <label htmlFor="voice-select">AI voice output</label>
            <select id="voice-select" value={voiceChoice} onChange={(event) => setVoiceChoice(event.target.value)}>
              {builtinVoices.map((voice) => (
                <option key={voice.choice_id || `builtin:${voice.id}`} value={voice.choice_id || `builtin:${voice.id}`}>
                  {voice.name} (clone sample)
                </option>
              ))}
              {SERVER_VOICE_OPTIONS.map((voice) => (
                <option key={voice.id} value={voice.id}>
                  {voice.label} (server)
                </option>
              ))}
              <option value="session_clone">Session cloned voice</option>
              {voiceProfiles.map((profile) => (
                <option key={profile.id} value={`profile:${profile.id}`}>
                  {profile.name} (profile)
                </option>
              ))}
              {voiceOptions.map((voice) => (
                <option key={voice.name} value={voice.name}>
                  {voice.name} ({voice.lang || "unknown"})
                </option>
              ))}
            </select>

            {selectedBuiltinVoice?.avatar_data_url ? (
              <div className="coach-voice-preview">
                <img src={selectedBuiltinVoice.avatar_data_url} alt={`${selectedBuiltinVoice.name} avatar`} />
                <div>
                  <strong>{selectedBuiltinVoice.name}</strong>
                  <p>Default clone-ready voice sample</p>
                </div>
              </div>
            ) : null}

            <div className="coach-voice-tools">
              <label htmlFor="voice-reference-upload">Reference sample</label>
              <input
                id="voice-reference-upload"
                type="file"
                accept="audio/*"
                disabled={isUploadingReference}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void uploadReferenceClip(file);
                  }
                  event.target.value = "";
                }}
              />
              {referenceClipTitle ? <p className="coach-side-note">Sample: {referenceClipTitle}</p> : null}
              <div className="coach-voice-profile-row">
                <input
                  type="text"
                  value={profileDraftName}
                  maxLength={100}
                  onChange={(event) => setProfileDraftName(event.target.value)}
                  placeholder="Profile name"
                />
                <button
                  type="button"
                  className="coach-btn coach-btn-small"
                  onClick={() => void createVoiceProfileFromReference()}
                  disabled={!referenceClipId || isCreatingProfile}
                >
                  {isCreatingProfile ? "Saving..." : "Save profile"}
                </button>
              </div>
            </div>

            {voiceProfiles.length ? (
              <div className="coach-voice-profiles-list">
                {voiceProfiles.map((profile) => (
                  <div key={profile.id} className="coach-voice-profile-item">
                    <span>{profile.name}</span>
                    <button
                      type="button"
                      className="coach-delete-btn"
                      onClick={() => void deleteVoiceProfile(profile.id)}
                      disabled={deletingProfileId === profile.id}
                      title="Delete profile"
                    >
                      <span className="material-symbols-rounded">
                        {deletingProfileId === profile.id ? "hourglass_top" : "delete"}
                      </span>
                    </button>
                  </div>
                ))}
              </div>
            ) : null}

            <label className="coach-checkbox-row" htmlFor="auto-speak-toggle">
              <input
                id="auto-speak-toggle"
                type="checkbox"
                checked={autoSpeakReplies}
                onChange={(event) => setAutoSpeakReplies(event.target.checked)}
              />
              Speak AI replies automatically
            </label>

            <button type="button" className="coach-btn" onClick={startConversation} disabled={!selectedSubject || !languageChoice}>
              Start conversation
            </button>
          </section>
        ) : (
          <section className="coach-session-shell">
            <div className="coach-meta-row">
              <div className="coach-chip">Subject: {activeSubject}</div>
              <div className="coach-chip">Language locked: {activeLanguage}</div>
              <div className="coach-chip">{username}</div>
              <div className="coach-chip">Turns: {turns.length}</div>
              <div className="coach-chip">Status: {activeSessionStatus}</div>
              <div className="coach-chip">Persona: {activePersona.label}</div>
            </div>

            <div className="coach-settings-shell">
              <div className="coach-settings-head">
                <div className="coach-settings-summary">
                  Persona: <strong>{activePersona.label}</strong> · Voice:{" "}
                  <strong>{selectedBuiltinVoice?.name || "Selected voice"}</strong>
                </div>
                <button
                  type="button"
                  className="coach-settings-toggle"
                  onClick={() => setShowSessionSettings((current) => !current)}
                  aria-expanded={showSessionSettings}
                  aria-controls="coach-session-settings"
                >
                  <span className="material-symbols-rounded">{showSessionSettings ? "expand_less" : "expand_more"}</span>
                  {showSessionSettings ? "Hide settings" : "Show settings"}
                </button>
              </div>

              {showSessionSettings ? (
                <div className="coach-inline-controls" id="coach-session-settings">
                  <label htmlFor="session-persona-select">Persona</label>
                  <select
                    id="session-persona-select"
                    value={activePersonaId}
                    onChange={(event) => {
                      const nextPersona = event.target.value;
                      setActivePersonaId(nextPersona);
                      setPersonaChoice(nextPersona);
                    }}
                  >
                    {PERSONA_OPTIONS.map((persona) => (
                      <option key={persona.id} value={persona.id}>
                        {persona.label}
                      </option>
                    ))}
                  </select>

                  <label htmlFor="session-voice-select">Voice</label>
                  <select
                    id="session-voice-select"
                    value={voiceChoice}
                    onChange={(event) => setVoiceChoice(event.target.value)}
                  >
                    {builtinVoices.map((voice) => (
                      <option key={voice.choice_id || `builtin:${voice.id}`} value={voice.choice_id || `builtin:${voice.id}`}>
                        {voice.name} (clone sample)
                      </option>
                    ))}
                    {SERVER_VOICE_OPTIONS.map((voice) => (
                      <option key={voice.id} value={voice.id}>
                        {voice.label} (server)
                      </option>
                    ))}
                    <option value="session_clone">Session cloned voice</option>
                    {voiceProfiles.map((profile) => (
                      <option key={profile.id} value={`profile:${profile.id}`}>
                        {profile.name} (profile)
                      </option>
                    ))}
                    {voiceOptions.map((voice) => (
                      <option key={voice.name} value={voice.name}>
                        {voice.name} ({voice.lang || "unknown"})
                      </option>
                    ))}
                  </select>

                  {selectedBuiltinVoice?.avatar_data_url ? (
                    <div className="coach-voice-preview">
                      <img src={selectedBuiltinVoice.avatar_data_url} alt={`${selectedBuiltinVoice.name} avatar`} />
                      <div>
                        <strong>{selectedBuiltinVoice.name}</strong>
                        <p>Using clone sample voice</p>
                      </div>
                    </div>
                  ) : null}

                  <label className="coach-checkbox-row" htmlFor="session-auto-speak-toggle">
                    <input
                      id="session-auto-speak-toggle"
                      type="checkbox"
                      checked={autoSpeakReplies}
                      onChange={(event) => setAutoSpeakReplies(event.target.checked)}
                    />
                    Auto-speak replies
                  </label>
                </div>
              ) : null}

              <div className="coach-view-toggle" role="tablist" aria-label="Conversation view">
                <button
                  type="button"
                  className={`coach-view-btn ${viewMode === "chat" ? "active" : ""}`}
                  onClick={() => setViewMode("chat")}
                >
                  Text chat
                </button>
                <button
                  type="button"
                  className={`coach-view-btn ${viewMode === "voice" ? "active" : ""}`}
                  onClick={() => setViewMode("voice")}
                >
                  Voice mode
                </button>
              </div>
            </div>

            {viewMode === "chat" ? (
              <section className="coach-thread" ref={threadRef}>
                <div className="coach-bubble ai">
                  <strong>Coach AI</strong>
                  <p>{starterQuestion}</p>
                </div>

                {turns.map((turn) => (
                  <div key={turn.id} className="coach-turn-pair">
                    <div className="coach-bubble user">
                      <strong>You</strong>
                      <p>{turn.transcript || "(No transcript)"}</p>
                    </div>
                    <div className="coach-bubble ai">
                      <strong>Coach AI {turn.score == null ? "" : `· ${Math.round(turn.score)}/100`}</strong>
                      <p>{turn.reply || "(No reply)"}</p>
                      {turn.reply ? (
                        <button
                          type="button"
                          className="coach-speak-btn"
                          onClick={() => void speakCoachText(turn.reply, { force: true })}
                        >
                          <span className="material-symbols-rounded">volume_up</span>
                          Replay voice
                        </button>
                      ) : null}
                      {turn.question ? <p className="coach-question">{turn.question}</p> : null}
                      {turn.correction ? <p className="coach-correction">Correction: {turn.correction}</p> : null}
                      {turn.feedback ? <p className="coach-feedback">{turn.feedback}</p> : null}
                      {turn.mistakes?.length ? (
                        <ul>
                          {turn.mistakes.slice(0, 3).map((mistake, index) => (
                            <li key={`${turn.id}-mistake-${index}`}>{mistake.detail || mistake.suggestion}</li>
                          ))}
                        </ul>
                      ) : null}
                      {turn.error ? <p className="coach-error">{turn.error}</p> : null}
                    </div>
                  </div>
                ))}

                {currentTurn ? (
                  <div className="coach-live-row">
                    <p>{isRecording ? "I'm listening..." : "Processing your answer..."}</p>
                    <div>{currentTurn.transcript || currentTurn.partialTranscript || "Listening..."}</div>
                  </div>
                ) : null}

                {!conversationEnded ? (
                  <div className="coach-text-compose">
                    <input
                      type="text"
                      value={textDraft}
                      onChange={(event) => setTextDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && !event.shiftKey) {
                          event.preventDefault();
                          void sendTextTurn();
                        }
                      }}
                      placeholder="Type a message instead of speaking..."
                      disabled={isSending || isEnding}
                    />
                    <button
                      type="button"
                      className="coach-btn coach-btn-small"
                      onClick={() => void sendTextTurn()}
                      disabled={isSending || isEnding || !String(textDraft || "").trim()}
                    >
                      Send
                    </button>
                  </div>
                ) : null}
              </section>
            ) : (
              <section className={`coach-voice-stage ${aiSpeakingNow ? "ai-speaking" : ""}`} aria-live="polite">
                <div className={`coach-voice-avatar-wrap ${aiSpeakingNow ? "is-speaking" : ""}`}>
                  {selectedBuiltinVoice?.avatar_data_url ? (
                    <img src={selectedBuiltinVoice.avatar_data_url} alt={`${selectedBuiltinVoice.name} avatar`} />
                  ) : (
                    <div className="coach-voice-avatar-fallback">{String(activePersona.label || "A").charAt(0)}</div>
                  )}
                </div>
                {voiceStageSnapshot.speaker === "ai" ? (
                  <div className={`coach-voice-wave ${aiSpeakingNow ? "active" : ""}`} aria-hidden="true">
                    {Array.from({ length: 14 }).map((_, index) => (
                      <span key={`wave-${index}`} style={{ "--bar-index": `${index}` }} />
                    ))}
                  </div>
                ) : null}
                <div className="coach-voice-stage-meta">
                  <strong>{voiceStageSnapshot.speaker === "ai" ? activePersona.label : "You"}</strong>
                  <span>
                    {voiceStageSnapshot.speaker === "ai"
                      ? (voiceStageSnapshot.status === "speaking" ? "Speaking now" : "Ready")
                      : (voiceStageSnapshot.status === "listening" ? "Listening now" : "Processing")}
                  </span>
                </div>
                <p className="coach-voice-stage-text">{voiceStageSnapshot.text}</p>
              </section>
            )}

            {!conversationEnded ? (
              <div className="coach-controls">
                <div className="coach-recording-text">{isRecording ? `I'm listening... ${recordingDuration}` : "Click mic to talk"}</div>
                <div className="coach-mic-wrap" role="presentation">
                  <div className={`coach-mic-halo ${isRecording ? "listening" : ""}`}>
                    <button
                      type="button"
                      className={`coach-mic-btn ${isRecording ? "recording" : ""}`}
                      onClick={handleMicToggle}
                      disabled={isSending || isEnding}
                      aria-label={isRecording ? "Stop recording" : "Start recording"}
                    >
                      <span className="material-symbols-rounded">mic</span>
                    </button>
                  </div>
                </div>

                <button
                  type="button"
                  className="coach-end-btn"
                  onClick={endConversation}
                  disabled={isSending || isRecording || isEnding}
                >
                  {isEnding ? "Ending..." : "End conversation"}
                </button>
              </div>
            ) : null}
          </section>
        )}

        {summary ? (
          <section className="coach-summary-card">
            <h2>Conversation Summary</h2>
            <p>
              Average score: <strong>{summary.average_score == null ? "N/A" : summary.average_score}</strong>
            </p>
            {summary.turn_count != null ? (
              <p>
                Scored turns:{" "}
                <strong>
                  {summary.scored_turn_count == null ? "0" : summary.scored_turn_count}/{summary.turn_count}
                </strong>
              </p>
            ) : null}
            <p>{summary.feedback_summary}</p>

            {Array.isArray(summary.improvement_points) && summary.improvement_points.length ? (
              <>
                <h3>Improve Next</h3>
                <ul>
                  {summary.improvement_points.map((item, index) => (
                    <li key={`improve-${index}`}>{item}</li>
                  ))}
                </ul>
              </>
            ) : null}

            <button type="button" className="coach-btn" onClick={startAnotherConversation}>
              Start another subject
            </button>
          </section>
        ) : null}

        {errorMessage ? <div className="coach-error-banner">{errorMessage}</div> : null}
      </main>
    </div>
  );
}

const root = createRoot(document.getElementById("coach-root"));
root.render(
  <AppErrorBoundary>
    <CoachApp />
  </AppErrorBoundary>,
);
