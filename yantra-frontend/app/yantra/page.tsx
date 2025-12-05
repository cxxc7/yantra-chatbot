"use client";

declare global {
  interface Window {
    SpeechRecognition?: { new (): SpeechRecognition } | undefined;
    webkitSpeechRecognition?: { new (): SpeechRecognition } | undefined;
  }

  interface SpeechRecognitionEvent {
    readonly resultIndex: number;
    readonly results: SpeechRecognitionResultList;
  }

  interface SpeechRecognitionResult {
    readonly isFinal: boolean;
    readonly 0: { transcript: string };
    readonly length: number;
    [index: number]: { transcript: string; confidence?: number };
  }

  interface SpeechRecognitionResultList {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }

  interface SpeechRecognition {
    lang: string;
    interimResults: boolean;
    maxAlternatives: number;
    onresult: ((ev: SpeechRecognitionEvent) => any) | null;
    onend: (() => any) | null;
    onerror: ((ev: any) => any) | null;
    start(): void;
    stop(): void;
    abort?(): void;
  }
}

import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: string;
}

interface ChatApiResponse {
  answer: string;
  used_context: string[];
  from_fallback: boolean;
  brochure_url?: string | null;
}

type Theme = "dark" | "light";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const SUGGESTED_QUESTIONS: string[] = [
  "What is the chisel diameter for Hyundai R30?",
  "Which breaker is compatible with SANY SY20?",
  "Compare JCB 3DX and Hyundai R30.",
  "Which is the best breaker option for Hyundai R30?",
];

const FOLLOW_UP_LINES: string[] = [
  "Anything else I can help you with?",
  "Want to compare any other models?",
  "Need details on spare parts or dealers next?",
];

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function formatTime(date: Date = new Date()): string {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function looksLikePipeTable(text: string): boolean {
  const lines = text.split("\n").map((l) => l.trim());
  const pipeLines = lines.filter((l) => l.includes("|"));
  if (pipeLines.length < 2) return false;
  for (const l of lines) {
    if (/^(\|?\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$/.test(l)) return true;
    if (/^-{3,}\s*\|/.test(l) || /\|\s*-{3,}/.test(l)) return true;
  }
  return false;
}

function boldParameterNames(content: string): string {
  if (!content) return content;
  const lines = content.split(/\n/).map((line) => {
    if (line.includes("|")) return line;
    if (/^\s*([-*]|\d+\.)\s/.test(line)) return line;
    return line.replace(
      /^([A-Za-z0-9 _/()&%-]+):\s*(.+)$/g,
      (_m, key, rest) => `**${key.trim()}**: ${rest.trim()}`
    );
  });
  return lines.join("\n");
}

function preprocessAssistant(content: string): string {
  if (!content) return content;
  let out = content;

  const fencedMatch = out.match(/^\s*```(?:\w+)?\n([\s\S]*?)\n```s*$/);
  if (fencedMatch) {
    out = fencedMatch[1];
  }

  const headerPatterns: RegExp[] = [
    /(Compatible machines:)(?!\n)/i,
    /(Here are the details[^\n]*?breaker model:)(?!\n)/i,
    /(Here are the details[^\n]*?breaker:)(?!\n)/i,
  ];
  headerPatterns.forEach((pat) => {
    out = out.replace(pat, (_m, g1) => `${g1}\n\n`);
  });

  out = out.replace(
    /(:\n[^\n]*?)(\b([A-Za-z][A-Za-z0-9 _/()&%-]{0,40}):\s)/g,
    (m, before, secondKey) => {
      return `${before}\n${secondKey}`;
    }
  );

  out = out.replace(/(Compatible machines:\n?)([^\n]+)/i, (m, head, list) => {
    const items = list
      .split(/[,;]+/)
      .map((s: string) => s.trim())
      .filter((v: string) => Boolean(v));
    if (items.length === 0) return m;
    const heading = "**COMPATIBLE MACHINES:**";
    const lines = items.map((i: string) => `- ${i}`);
    return `${heading}\n\n${lines.join("\n")}`;
  });

  out = out.replace(/(^|\n)Compatible machines:\s*\n+/i, (m) => {
    return `${m.startsWith("\n") ? "\n" : ""}**COMPATIBLE MACHINES:**\n\n`;
  });

  const lines = out.split(/\n/);
  let inCompat = false;
  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx];
    if (/^\*\*COMPATIBLE MACHINES:\*\*$/.test(line.trim())) {
      inCompat = true;
      continue;
    }
    if (inCompat) {
      if (!line.trim()) continue;
      if (/^\*\*.+\*\*$/.test(line.trim()) || /^[A-Za-z0-9 _/()&%-]+:\s/.test(line)) {
        if (!/^\s*[-]/.test(line.trim()) && !/^Machine:\s/.test(line.trim())) inCompat = false;
        continue;
      }
      if (/^[-*]\s+/.test(line)) {
        const name = line.replace(/^[-*]\s+/, "").trim();
        lines[idx] = `- ${name}`;
        continue;
      }
      if (!/^-/.test(line) && /^[A-Za-z0-9 .()/-]{2,}$/.test(line.trim())) {
        lines[idx] = `- ${line.trim()}`;
        continue;
      }
      inCompat = false;
    }
  }
  out = lines.join("\n");

  const allLines = out.split("\n");
  let i = 0;
  const newLines: string[] = [];
  while (i < allLines.length) {
    if (allLines[i].includes("|")) {
      const start = i;
      let end = i;
      while (end + 1 < allLines.length && allLines[end + 1].includes("|")) end++;
      const block = allLines.slice(start, end + 1).join("\n");
      if (looksLikePipeTable(block)) {
        if (newLines.length > 0 && newLines[newLines.length - 1].trim() !== "") {
          newLines.push("");
        }
        const cleanedBlock = block
          .replace(/^\s*```/g, "")
          .replace(/```\s*$/g, "");
        newLines.push(cleanedBlock);
        if (end + 1 < allLines.length && allLines[end + 1].trim() !== "") {
          newLines.push("");
        }
        i = end + 1;
        continue;
      } else {
        newLines.push(allLines[i]);
        i++;
        continue;
      }
    } else {
      newLines.push(allLines[i]);
      i++;
    }
  }

  out = newLines.join("\n");

  return out;
}

export default function YantraChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [greeting, setGreeting] = useState("Hello");
  const [theme, setTheme] = useState<Theme>("dark");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // microphone / speech recognition state
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  // base value of textarea when recording started
  const baseBeforeRecordingRef = useRef<string>("");
  // current interim visible text (not committed to textarea until final)
  const [interimText, setInterimText] = useState<string>("");
  // flag when permission denied
  const [micPermissionDenied, setMicPermissionDenied] = useState<boolean>(false);

  // server recording state (MediaRecorder)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [isServerRecording, setIsServerRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);

  // client-only mount + support detection
  const [isMounted, setIsMounted] = useState(false);
  const [supportsSpeechRecognition, setSupportsSpeechRecognition] = useState(false);

  // language selector: 'en' | 'hi' | 'kn'
  const [language, setLanguage] = useState<"en" | "hi" | "kn">("en");
  // force server STT override
  const [forceServerStt, setForceServerStt] = useState<boolean>(false);

  // brochure modal state
  const [brochureOpen, setBrochureOpen] = useState(false);
  const [brochureUrl, setBrochureUrl] = useState<string | null>(null);

  // refs to control scrolling
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setGreeting(getGreeting());
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem("yantra-theme");
    if (stored === "dark" || stored === "light") {
      setTheme(stored);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem("yantra_chat_messages_v1");
      if (raw) {
        const parsed = JSON.parse(raw) as Message[];
        if (Array.isArray(parsed)) {
          setMessages(parsed);
        }
      }
    } catch (e) {
      console.warn("Failed to restore messages", e);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem("yantra_chat_messages_v1", JSON.stringify(messages));
    } catch (e) {
      console.warn("Failed to save messages", e);
    }
  }, [messages]);

  useEffect(() => {
    if (messagesContainerRef.current) {
      setTimeout(() => {
        try {
          messagesContainerRef.current!.scrollTo({
            top: messagesContainerRef.current!.scrollHeight,
            behavior: "smooth",
          });
        } catch {
          messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }
      }, 50);
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const quickActions = useMemo(() => SUGGESTED_QUESTIONS, []);

  const toggleTheme = () => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      if (typeof window !== "undefined") {
        window.localStorage.setItem("yantra-theme", next);
      }
      return next;
    });
  };

  const handleClearChat = () => {
    setMessages([]);
    setInput("");
    textareaRef.current?.focus();
    try {
      if (typeof window !== "undefined") {
        window.localStorage.removeItem("yantra_chat_messages_v1");
      }
    } catch {}
  };

  const handleCopy = (content: string) => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard.writeText(content).catch((err) => {
      console.error("Failed to copy", err);
    });
  };

  // open brochure modal (normalize href first)
  const openBrochureModal = (href: string) => {
    try {
      const u = new URL(href, window.location.href);
      let final = u.href;
      if (u.pathname.startsWith("/brochures/view/") || u.pathname.startsWith("/brochures/raw/") || u.pathname.includes("/brochures/")) {
        if (!final.includes("#")) final = `${final}#toolbar=1`;
      }
      setBrochureUrl(final);
      setBrochureOpen(true);
    } catch (e) {
      setBrochureUrl(href);
      setBrochureOpen(true);
    }
  };

  const closeBrochureModal = () => {
    setBrochureOpen(false);
    setTimeout(() => setBrochureUrl(null), 200);
  };

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    const now = formatTime();
    const lc = trimmed.toLowerCase();

    if (lc.includes("thank you") || lc.includes("thanks")) {
      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
        timestamp: now,
      };
      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          "You're welcome! Anything else I can help you with today?",
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, userMsg, botMsg]);
      setInput("");
      textareaRef.current?.focus();
      return;
    }

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      timestamp: now,
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const backendMessages = newMessages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages: backendMessages }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Chat request failed");
      }

      const data: ChatApiResponse = await res.json();

      let combined = data.answer.trim();
      const followup = randomFrom(FOLLOW_UP_LINES);
      combined = `${combined}\n\n_${followup}_`;

      // Append the brochure link in the message, but DO NOT auto-open the modal.
      if (data.brochure_url) {
        combined = `${combined}\n\n[📘 Open Brochure](${data.brochure_url})`;
      }

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: combined,
        timestamp: formatTime(),
      };

      setMessages((prev) => [...prev, botMsg]);

    } catch (err: any) {
      console.error(err);
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Error: ${err.message || "Something went wrong"}`,
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleSend = async () => {
    if (loading) return;
    await sendMessage(input);
  };

  const handleQuickQuestion = (q: string) => {
    if (loading) return;
    setInput("");
    void sendMessage(q);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading) {
        void handleSend();
      }
    }
  };

  const hasMessages = messages.length > 0;

  const rootClass =
    theme === "dark"
      ? "min-h-screen flex flex-col bg-slate-950 text-slate-100"
      : "min-h-screen flex flex-col bg-slate-50 text-slate-900";

  const headerBorder =
    theme === "dark" ? "border-slate-800" : "border-slate-200";

  const headerBg =
    theme === "dark"
      ? "bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950"
      : "bg-white";

  const headerSubText =
    theme === "dark" ? "text-slate-400" : "text-slate-500";

  const cardClass =
    theme === "dark"
      ? "rounded-2xl border border-slate-800/80 bg-slate-900/60 shadow-[0_20px_60px_rgba(0,0,0,0.6)]"
      : "rounded-2xl border border-slate-200 bg-white shadow-[0_20px_40px_rgba(15,23,42,0.12)]";

  const dividerBorder =
    theme === "dark" ? "border-slate-800/80" : "border-slate-200";

  const assistantBubbleClass =
    theme === "dark"
      ? "bg-slate-800/90 text-slate-100 border border-slate-700/60"
      : "bg-slate-100 text-slate-900 border border-slate-200";

  const timestampClass =
    theme === "dark" ? "text-slate-500" : "text-slate-400";

  const inputBorder =
    theme === "dark" ? "border-slate-800" : "border-slate-300";

  const inputBg =
    theme === "dark" ? "bg-slate-900/80" : "bg-white";

  // ------- Speech recognition helpers -------
  useEffect(() => {
    // mark mounted and run feature detection on client only
    setIsMounted(true);
    const supported = typeof window !== "undefined" && (!!(window as any).SpeechRecognition || !!(window as any).webkitSpeechRecognition);
    setSupportsSpeechRecognition(Boolean(supported));
  }, []);

  useEffect(() => {
    if (!supportsSpeechRecognition) return;

    const SpeechRecognitionConstructor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    const rec: SpeechRecognition = new SpeechRecognitionConstructor();
    rec.lang = "en-IN";
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    recognitionRef.current = rec;

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      // Build interim + final from results. We DO NOT mutate the textarea's value for interim.
      let interim = "";
      let final = "";
      for (let i = 0; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) {
          final += r[0].transcript;
        } else {
          interim += r[0].transcript;
        }
      }

      setInterimText(interim || "");

      // If final result present, commit to textarea `input` (append to whatever is currently in textarea)
      if (final && final.trim()) {
        setInput((prev) => {
          const appended = (prev + (prev && !prev.endsWith(" ") ? " " : "") + final).trim();
          return appended + " ";
        });
        baseBeforeRecordingRef.current = "";
        setInterimText("");
      }
    };

    rec.onend = () => {
      setIsRecording(false);
      setInterimText("");
      baseBeforeRecordingRef.current = "";
    };

    rec.onerror = (err: any) => {
      console.error("SpeechRecognition error", err);
      try {
        const name = err?.error || err?.name || "";
        if (typeof name === "string" && (name.toLowerCase().includes("not-allowed") || name.toLowerCase().includes("denied"))) {
          setMicPermissionDenied(true);
        }
      } catch {}
      setIsRecording(false);
      setInterimText("");
      baseBeforeRecordingRef.current = "";
    };

    return () => {
      try {
        rec.onresult = null;
        rec.onend = null;
        rec.onerror = null;
        recognitionRef.current = null;
      } catch {}
    };
  // only re-run when supportsSpeechRecognition flips
  }, [supportsSpeechRecognition]);

  // Decide recording strategy: automatic
  const shouldUseServerStt = (lang: string, forceServer: boolean) => {
    // Auto rules:
    // - If forceServer true => server
    // - If Kannada (kn) => server (browser STT rarely supports)
    // - Else if browser supports SpeechRecognition => use it
    // - else => server
    if (forceServer) return true;
    if (lang === "kn") return true;
    if (supportsSpeechRecognition) return false;
    return true;
  };

  // -------- Server recording (MediaRecorder) flow --------
  const startServerRecording = async () => {
    try {
      setMicPermissionDenied(false);
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mr;
      setIsServerRecording(true);
      setIsTranscribing(false);
      audioChunksRef.current = [];

      mr.ondataavailable = (ev: BlobEvent) => {
        if (ev.data && ev.data.size > 0) {
          audioChunksRef.current.push(ev.data);
        }
      };

      mr.onstop = async () => {
        setIsServerRecording(false);
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        // send to server
        await handleServerUpload(blob);
        // stop the tracks to release mic
        try {
          stream.getTracks().forEach((t) => t.stop());
        } catch {}
      };

      mr.onerror = (ev) => {
        console.error("MediaRecorder error", ev);
        setIsServerRecording(false);
        try {
          stream.getTracks().forEach((t) => t.stop());
        } catch {}
      };

      mr.start();
    } catch (e: any) {
      console.error("startServerRecording:", e);
      if (String(e).toLowerCase().includes("permission")) {
        setMicPermissionDenied(true);
      }
      setIsServerRecording(false);
    }
  };

  const stopServerRecording = () => {
    try {
      mediaRecorderRef.current?.stop();
    } catch (e) {
      console.warn("stopServerRecording:", e);
      setIsServerRecording(false);
    }
  };

  const handleServerUpload = async (blob: Blob) => {
    setIsTranscribing(true);
    setIsRecording(false);
    setInterimText("");
    baseBeforeRecordingRef.current = "";

    // Upload as form-data; expects backend /api/stt?lang=en|hi|kn
    const fd = new FormData();
    fd.append("file", blob, "voice.webm");

    try {
      const res = await fetch(`${API_BASE_URL}/api/stt?lang=${language}`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || `STT upload failed (${res.status})`);
      }
      const json = await res.json().catch(() => ({}));
      const transcript = (json && (json.transcript || json.text || json.result)) || "";
      if (transcript && transcript.trim()) {
        // append transcript to input
        setInput((prev) => {
          const base = prev || "";
          const sep = base && !base.endsWith(" ") ? " " : "";
          return `${base}${sep}${transcript}`.trim() + " ";
        });
      } else {
        // if server returned nothing
        console.warn("Empty transcript from server", json);
      }
    } catch (err: any) {
      console.error("handleServerUpload error", err);
      alert("Failed to transcribe audio: " + (err?.message || "unknown"));
    } finally {
      setIsTranscribing(false);
      // ensure mediaRecorder cleaned up
      try {
        mediaRecorderRef.current = null;
        audioChunksRef.current = [];
      } catch {}
      textareaRef.current?.focus();
    }
  };

  // Start recognition (either browser or server) depending on strategy
  const startRecognition = async () => {
    const useServer = shouldUseServerStt(language, forceServerStt);

    // Snapshot base text so overlay can show it
    baseBeforeRecordingRef.current = input || "";

    if (useServer) {
      // server recording path
      await startServerRecording();
      return;
    }

    // browser SpeechRecognition path
    if (!supportsSpeechRecognition || !recognitionRef.current) {
      setMicPermissionDenied(true);
      return;
    }

    try {
      setInterimText("");
      setIsRecording(true);
      setMicPermissionDenied(false);
      // set recognition language (map our language codes to locale)
      let langToUse = "en-IN";
      if (language === "hi") langToUse = "hi-IN";
      if (language === "kn") langToUse = "kn-IN"; // fallback; still we'll prefer server for kn
      recognitionRef.current.lang = langToUse;
      try {
        recognitionRef.current!.start();
      } catch (e: any) {
        console.warn("recognition.start() failed:", e);
        if (String(e).toLowerCase().includes("not allowed") || String(e).toLowerCase().includes("permission")) {
          setMicPermissionDenied(true);
        }
        setIsRecording(false);
      }
      textareaRef.current?.focus();
    } catch (e) {
      console.error("Failed starting recognition", e);
      setIsRecording(false);
    }
  };

  const stopRecognition = () => {
    // stop both possible flows
    try {
      if (isServerRecording) {
        stopServerRecording();
      }
    } catch {}
    try {
      recognitionRef.current?.stop();
    } catch {}
    setIsRecording(false);
    setInterimText("");
    baseBeforeRecordingRef.current = "";
  };

  const toggleMic = async () => {
    // if currently transcribing, ignore toggles
    if (isTranscribing) return;

    // If server recording active -> stop
    if (isServerRecording) {
      stopServerRecording();
      return;
    }

    // If browser recognition active -> stop
    if (isRecording) {
      stopRecognition();
      return;
    }

    // start fresh
    await startRecognition();
  };

  // ensure we stop recognition when unmounting
  useEffect(() => {
    return () => {
      try {
        if (recognitionRef.current) recognitionRef.current.abort?.();
      } catch {}
      try {
        if (mediaRecorderRef.current) mediaRecorderRef.current.stop();
      } catch {}
    };
  }, []);

  // Small helper: mic SVG component
  const MicSVG = ({ recording }: { recording: boolean }) => (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`${recording ? "text-rose-500" : "text-slate-500"} w-4 h-4`}
      role="img"
      focusable="false"
    >
      <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M19 11v1a7 7 0 0 1-14 0v-1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 19v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );

  // Helper label to show small status next to mic
  const micStatusLabel = () => {
    if (isTranscribing) return "Transcribing...";
    if (isServerRecording) return "Recording (server)";
    if (isRecording) return "Listening";
    return "Mic";
  };

  return (
    <div className={rootClass}>
      {/* Accessible status live region for mic state */}
      <div aria-live="polite" className="sr-only" role="status">
        {isRecording || isServerRecording || isTranscribing ? "Microphone active" : "Microphone inactive"}
      </div>

      {/* Header */}
      <header
        className={`border-b px-4 py-3 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between ${headerBorder} ${headerBg}`}
      >
        <div className="flex flex-col">
          <h1 className="text-xl font-semibold tracking-tight">
            YantraBuddy
          </h1>
          <p className={`text-xs ${headerSubText}`}>
            Your source for everything on rock breakers, spare parts, and dealer insights.
          </p>
        </div>

        <div className="flex items-center gap-3 mt-1 sm:mt-0 text-[11px]">
          <div className="flex items-center gap-2">
            {/* Language selector (client-only to avoid SSR mismatch) */}
            {isMounted && (
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value as "en" | "hi" | "kn")}
                className="text-xs px-2 py-1 rounded-full border bg-white dark:bg-slate-800"
                title="Language for speech"
              >
                <option value="en">English</option>
                <option value="hi">Hindi</option>
                <option value="kn">Kannada</option>
              </select>
            )}

            {/* Force server STT toggle */}
            {isMounted && (
              <label className="text-xs inline-flex items-center gap-2 px-2 py-1 rounded-full border bg-white dark:bg-slate-800">
                <input
                  type="checkbox"
                  checked={forceServerStt}
                  onChange={(e) => setForceServerStt(e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-[11px]">Server STT</span>
              </label>
            )}
          </div>

          <button
            type="button"
            onClick={toggleTheme}
            className="px-3 py-2 rounded-full border border-slate-500/40 bg-slate-900/10 text-xs hover:bg-slate-900/20 transition"
          >
            {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
          </button>
          <button
            type="button"
            onClick={handleClearChat}
            className="px-3 py-2 rounded-full border border-red-500/60 bg-red-500/10 text-xs text-red-400 hover:bg-red-500/20 transition"
          >
            🧹 Clear
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col max-w-4xl mx-auto w-full p-4">
        <section
          className={`${cardClass} w-full flex flex-col h-[82vh]`}
        >
          {/* Greeting + quick actions */}
          <div className={`p-4 border-b ${dividerBorder}`}>
            <p className="text-sm font-medium">
              {greeting}, welcome to YantraLive.
            </p>
            <p className="text-xs mt-1 text-slate-400">
              {hasMessages
                ? "What else would you like to check? You can also try these questions:"
                : "How can I help you today? Here are some common questions:"}
            </p>

            <div className="mt-3 flex flex-wrap gap-2">
              {quickActions.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => handleQuickQuestion(q)}
                  className="text-xs px-3 py-1 rounded-full border border-sky-500/60 bg-sky-900/40 hover:bg-sky-700/60 hover:border-sky-400 transition shadow-sm"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Messages area */}
          <div
            ref={messagesContainerRef}
            className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3"
          >
            {!hasMessages && (
              <div className="text-lg font-medium text-slate-300 text-center mt-20">
                Ask anything about breakers, machine compatibility, spare
                parts, or dealers.
              </div>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`flex flex-col max-w-[80%] ${m.role === "user" ? "items-end" : "items-start"}`}>
                  <div className={`rounded-2xl px-3 py-2 text-sm whitespace-pre-wrap ${m.role === "user" ? "bg-sky-600 text-white shadow-md" : assistantBubbleClass}`}>
                    {m.role === "assistant" ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          table: ({ children }) => (
                            <div className="overflow-x-auto my-2">
                              <table className="text-xs border-collapse w-full">
                                {children}
                              </table>
                            </div>
                          ),
                          th: (props) => (
                            <th
                              {...props}
                              className="border border-slate-600 px-2 py-1 bg-slate-900 text-left"
                            />
                          ),
                          td: (props) => (
                            <td
                              {...props}
                              className="border border-slate-700 px-2 py-1 align-top"
                            />
                          ),
                          ul: (props) => (
                            <ul
                              {...props}
                              className="list-disc list-inside space-y-1"
                            />
                          ),
                          a: (props) => {
                            const href = String(props.href || "");
                            try {
                              const u = new URL(href, window.location.href);
                              if (u.pathname.startsWith("/brochures") || href.includes("/brochures/")) {
                                return (
                                  <a
                                    {...props}
                                    href="#"
                                    onClick={(e) => {
                                      e.preventDefault();
                                      openBrochureModal(href);
                                    }}
                                  >
                                    {props.children}
                                  </a>
                                );
                              }
                            } catch {
                              // fall back
                            }
                            return (
                              <a {...props} target="_blank" rel="noopener noreferrer">
                                {props.children}
                              </a>
                            );
                          },
                        }}
                      >
                        {boldParameterNames(preprocessAssistant(m.content))}
                      </ReactMarkdown>
                    ) : (
                      m.content
                    )}
                  </div>

                  <div className="mt-1 flex items-center justify-between w-full gap-2">
                    <span className={`text-[10px] ${timestampClass}`}>
                      {m.timestamp}
                    </span>

                    {m.role === "assistant" && (
                      <div className="flex items-center gap-1 text-[10px] text-slate-500">
                        <button
                          type="button"
                          onClick={() => handleCopy(m.content)}
                          className="px-1 py-[1px] rounded-full border border-slate-600/60 hover:bg-slate-800/60"
                          title="Copy reply"
                        >
                          📋
                        </button>
                        <button
                          type="button"
                          className="px-1 py-[1px] rounded-full border border-slate-600/60 hover:bg-slate-800/60"
                          title="Like"
                        >
                          👍
                        </button>
                        <button
                          type="button"
                          className="px-1 py-[1px] rounded-full border border-slate-600/60 hover:bg-slate-800/60"
                          title="Dislike"
                        >
                          👎
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            <div ref={messagesEndRef} />
            {loading && (
              <div className="flex justify-start">
                <div className="text-xs text-slate-400 px-3 py-2 bg-slate-800 rounded-xl border border-slate-700/60">
                  Thinking based on the YantraLive datasets…
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <form
            className={`border-t p-3 flex gap-2 rounded-b-2xl flex-shrink-0 ${dividerBorder} ${theme === "dark" ? "bg-slate-950/70" : "bg-slate-50"}`}
            onSubmit={(e) => {
              e.preventDefault();
              if (!loading) void handleSend();
            }}
          >
            {/* input area wrapper: relative so we can overlay interim text */}
            <div className="relative flex-1">
              {/* Actual textarea - while recording we make text transparent and show overlay instead */}
              <textarea
                ref={textareaRef}
                className={`w-full text-sm rounded-xl border ${inputBorder} ${inputBg} p-2 resize-none outline-none focus:ring-1 focus:ring-sky-500 focus:border-sky-500 ${ (isRecording || isServerRecording) ? "text-transparent caret-black" : "" }`}
                rows={1}
                placeholder="Type your question about breakers, machines, spare parts, dealers, pricing..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                aria-label="Message input"
                style={{ whiteSpace: "pre-wrap" }}
              />

              {/* Overlay showing baseBeforeRecording + interim while recording */}
              {(isRecording || isServerRecording) && (
                <div
                  aria-hidden
                  className="absolute inset-0 pointer-events-none px-2 py-2 overflow-hidden"
                  style={{
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    display: "flex",
                    alignItems: "flex-start",
                  }}
                >
                  <div className="w-full text-sm leading-[1.3]">
                    {/* show the base text snapshot (if any) */}
                    <span className="text-slate-200">{baseBeforeRecordingRef.current}</span>
                    {/* show the interim in grey which updates live for browser STT; for server recording show a small dot + timer text */}
                    {isRecording ? (
                      <span className="text-slate-400">{interimText}</span>
                    ) : (
                      <span className="text-slate-400"> ● Recording... </span>
                    )}
                    {/* show a caret indicator */}
                    <span className="inline-block w-1 h-4 align-middle bg-sky-400 ml-1 animate-pulse" />
                  </div>
                </div>
              )}

              {/* Microphone permission hint */}
              {micPermissionDenied && (
                <div className="absolute -bottom-5 left-0 text-xs text-rose-400">
                  Microphone access required — please allow the browser to use your microphone.
                </div>
              )}
            </div>

            {/* Mic button - render an initial server-safe placeholder; after mount replace with interactive button */}
            {!isMounted ? (
              // server-rendered placeholder: disabled, fixed title — avoids hydration mismatch
              <button
                type="button"
                title="Speech recognition not supported"
                aria-pressed={false}
                className="px-3 py-2 rounded-xl border border-slate-300 hover:bg-slate-100 disabled:opacity-50 transition self-end"
                disabled
              >
                <span className="inline-flex items-center gap-2 text-sm">
                  <span aria-hidden>
                    {/* placeholder mic SVG (static) */}
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-slate-500" role="img" focusable="false">
                      <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M19 11v1a7 7 0 0 1-14 0v-1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M12 19v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </span>
                  <span className="text-xs">Mic</span>
                </span>
              </button>
            ) : (
              // interactive mic button (client-side only after mount)
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={toggleMic}
                  title={ supportsSpeechRecognition ? (isRecording || isServerRecording ? "Stop recording" : "Start recording") : "Speech recognition not supported; server mode will be used" }
                  aria-pressed={isRecording || isServerRecording}
                  className={`px-3 py-2 rounded-xl border ${ (isRecording || isServerRecording) ? "border-rose-400 bg-rose-600/10 ring-2 ring-rose-500/30 transform-gpu scale-100 animate-pulse-slow" : "border-slate-300" } hover:bg-slate-100 disabled:opacity-50 transition self-end`}
                  disabled={isTranscribing}
                  aria-label={ (isRecording || isServerRecording) ? "Stop microphone" : "Start microphone" }
                >
                  <span className="inline-flex items-center gap-2 text-sm">
                    <span aria-hidden className="flex items-center">
                      <MicSVG recording={isRecording || isServerRecording} />
                    </span>
                    <span className="text-xs">{micStatusLabel()}</span>
                  </span>
                </button>

                {/* show small transcribing loader if uploading/processing */}
                {isTranscribing && (
                  <div className="text-xs px-2 py-1 rounded-md border bg-white dark:bg-slate-800">
                    Transcribing...
                  </div>
                )}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="text-sm px-4 py-2 rounded-xl border border-sky-500 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed transition self-end shadow-md"
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </section>
      </main>

      {/* Brochure modal */}
      {brochureOpen && brochureUrl && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center p-6"
          aria-modal="true"
          role="dialog"
        >
          <div className="absolute inset-0 bg-black/60" onClick={closeBrochureModal} />
          <div className="relative w-full max-w-4xl h-[85vh] bg-white dark:bg-slate-900 rounded-xl overflow-hidden shadow-2xl z-50">
            <div className="flex items-center justify-between gap-4 p-3 border-b">
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold">Brochure preview</span>
                <span className="text-xs text-slate-500">{decodeURIComponent(brochureUrl.split("/").pop() || "")}</span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={brochureUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs px-3 py-1 rounded border border-slate-300 hover:bg-slate-100"
                >
                  Open in new tab
                </a>
                <a
                  href={brochureUrl}
                  download
                  className="text-xs px-3 py-1 rounded border border-slate-300 hover:bg-slate-100"
                >
                  Download
                </a>
                <button
                  onClick={closeBrochureModal}
                  className="text-sm px-3 py-1 rounded border border-red-300 text-red-500 hover:bg-red-50"
                >
                  Close
                </button>
              </div>
            </div>

            <div className="w-full h-[calc(100%-56px)]">
              <iframe
                src={brochureUrl}
                title="Brochure preview"
                className="w-full h-full"
                sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}