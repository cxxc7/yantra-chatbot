"use client";

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

// helper: detect if a block of lines looks like a pipe-table
function looksLikePipeTable(text: string): boolean {
  // simple heuristic: at least two lines containing '|' and a header separator line with --- or ---|---
  const lines = text.split("\n").map((l) => l.trim());
  const pipeLines = lines.filter((l) => l.includes("|"));
  if (pipeLines.length < 2) return false;
  // check for a separator row like |---|---| or ---|--- pattern
  for (const l of lines) {
    if (/^(\|?\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$/.test(l)) return true;
    if (/^-{3,}\s*\|/.test(l) || /\|\s*-{3,}/.test(l)) return true;
  }
  return false;
}

// helper: bold parameter names in assistant content (skip table rows)
function boldParameterNames(content: string): string {
  if (!content) return content;
  const lines = content.split(/\n/).map((line) => {
    // If the line looks like a table row (contains |) - skip bolding
    if (line.includes("|")) return line;
    // Skip bullets or numbered lists to avoid breaking markdown list formatting
    if (/^\s*([-*]|\d+\.)\s/.test(line)) return line;
    // Bold "Key: value" patterns
    return line.replace(
      /^([A-Za-z0-9 _/()&%-]+):\s*(.+)$/g,
      (_m, key, rest) => `**${key.trim()}**: ${rest.trim()}`
    );
  });
  return lines.join("\n");
}

// helper: ensure certain headers start on their own line and split concatenated key:value pairs
function preprocessAssistant(content: string): string {
  if (!content) return content;

  let out = content;

  // Remove surrounding fenced code blocks if the whole message is wrapped (common when backend accidentally fences)
  // But be careful: only strip if it's a single fenced block surrounding entire content.
  const fencedMatch = out.match(/^\s*```(?:\w+)?\n([\s\S]*?)\n```s*$/);
  if (fencedMatch) {
    out = fencedMatch[1];
  }

  // Force newline after these headers if followed by text
  const headerPatterns: RegExp[] = [
    /(Compatible machines:)(?!\n)/i,
    /(Here are the details[^\n]*?breaker model:)(?!\n)/i,
    /(Here are the details[^\n]*?breaker:)(?!\n)/i,
  ];
  headerPatterns.forEach((pat) => {
    out = out.replace(pat, (_m, g1) => `${g1}\n\n`); // blank line after header for markdown list separation
  });

  // If a line contains two key:value pairs back to back, insert newline before second key
  // Example: "Breaker weight: 180 KG Price: 10000 INR" -> split before "Price:"
  out = out.replace(
    /(:\n[^\n]*?)(\b([A-Za-z][A-Za-z0-9 _/()&%-]{0,40}):\s)/g,
    (m, before, secondKey) => {
      return `${before}\n${secondKey}`;
    }
  );

  // Normalize "Compatible machines" section: uppercase bold heading, convert lists to bullets
  // 1. Single-line list form: "Compatible machines: A, B" -> expand to bullets
  out = out.replace(/(Compatible machines:\n?)([^\n]+)/i, (m, head, list) => {
    const items = list
      .split(/[,;]+/)
      .map((s: string) => s.trim())
      .filter((v: string) => Boolean(v));
    if (items.length === 0) return m;
    const heading = "**COMPATIBLE MACHINES:**"; // bold uppercase heading
    const lines = items.map((i: string) => `- ${i}`);
    return `${heading}\n\n${lines.join("\n")}`;
  });

  // 2. Multi-line bullet form following heading: transform heading then bullets.
  out = out.replace(/(^|\n)Compatible machines:\s*\n+/i, (m) => {
    return `${m.startsWith("\n") ? "\n" : ""}**COMPATIBLE MACHINES:**\n\n`;
  });

  // Convert lines starting with * or - and not already parameter format after the compatible machines heading into bullets.
  const lines = out.split(/\n/);
  let inCompat = false;
  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx];
    if (/^\*\*COMPATIBLE MACHINES:\*\*$/.test(line.trim())) {
      inCompat = true;
      continue;
    }
    if (inCompat) {
      if (!line.trim()) {
        // blank lines allowed; keep and continue
        continue;
      }
      // Stop section if we hit another bold heading or a parameter line not a machine list
      if (/^\*\*.+\*\*$/.test(line.trim()) || /^[A-Za-z0-9 _/()&%-]+:\s/.test(line)) {
        if (!/^\s*[-]/.test(line.trim()) && !/^Machine:\s/.test(line.trim())) inCompat = false;
        continue;
      }
      if (/^[-*]\s+/.test(line)) {
        const name = line.replace(/^[-*]\s+/, "").trim();
        lines[idx] = `- ${name}`;
        continue;
      }
      // Plain line that does not start with bullet; assume still machine list if simple token
      if (!/^-/.test(line) && /^[A-Za-z0-9 .()/-]{2,}$/.test(line.trim())) {
        lines[idx] = `- ${line.trim()}`;
        continue;
      }
      inCompat = false;
    }
  }
  out = lines.join("\n");

  // Ensure pipe tables are surrounded by blank lines so remark-gfm picks them up
  // We'll scan for contiguous groups of lines that contain pipes and wrap them with blank lines
  const allLines = out.split("\n");
  let i = 0;
  const newLines: string[] = [];
  while (i < allLines.length) {
    // gather a potential table block starting at i
    if (allLines[i].includes("|")) {
      // peek ahead to collect contiguous pipe-lines
      const start = i;
      let end = i;
      while (end + 1 < allLines.length && allLines[end + 1].includes("|")) end++;
      const block = allLines.slice(start, end + 1).join("\n");
      if (looksLikePipeTable(block)) {
        // ensure there's a blank line before and after
        if (newLines.length > 0 && newLines[newLines.length - 1].trim() !== "") {
          newLines.push("");
        }
        // remove any accidental leading/trailing backticks around the block lines
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
        // not a table, push current line and continue
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

  // restore chat history on mount
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

  // persist messages to localStorage whenever messages change
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem("yantra_chat_messages_v1", JSON.stringify(messages));
    } catch (e) {
      console.warn("Failed to save messages", e);
    }
  }, [messages]);

  // scroll to bottom whenever messages change
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

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    const now = formatTime();
    const lc = trimmed.toLowerCase();

    // Only say "You're welcome" when user explicitly thanks
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

      // Append a short "anything else" line to the same answer
      const followup = randomFrom(FOLLOW_UP_LINES);
      const combinedAnswer = `${data.answer.trim()}\n\n_${followup}_`;

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: combinedAnswer,
        timestamp: formatTime(),
      };

      setMessages((prev) => [...prev, botMsg]);
      textareaRef.current?.focus();
    } catch (err: any) {
      console.error(err);
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Error: ${err.message || "Something went wrong"}`,
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      textareaRef.current?.focus();
    } finally {
      setLoading(false);
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

  return (
    <div className={rootClass}>
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

            {/* FAQ quick questions */}
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
                        }}
                      >
                        {boldParameterNames(preprocessAssistant(m.content))}
                      </ReactMarkdown>
                    ) : (
                      m.content
                    )}
                  </div>

                  {/* Timestamp + tiny actions */}
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
            <textarea
              ref={textareaRef}
              className={`flex-1 text-sm rounded-xl border ${inputBorder} ${inputBg} p-2 resize-none outline-none focus:ring-1 focus:ring-sky-500 focus:border-sky-500`}
              rows={1}
              placeholder="Type your question about breakers, machines, spare parts, dealers, pricing..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
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
    </div>
  );
}
