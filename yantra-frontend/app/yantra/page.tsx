"use client";

import React, { useState } from "react";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
}

interface ChatApiResponse {
  answer: string;
  used_context: string[];
  from_fallback: boolean;
}

const API_BASE_URL = "http://127.0.0.1:8000";
export default function YantraChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
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

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err: any) {
      console.error(err);
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Error: ${err.message || "Something went wrong"}`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading) {
        handleSend();
      }
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-4 py-3 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col">
          <h1 className="text-lg font-semibold">
            YantraLive End-Customer Assistant
          </h1>
          <p className="text-xs text-slate-400">
            Answers only from the latest end-customer rock breaker dataset.
          </p>
        </div>
        <div className="text-[11px] text-slate-500 mt-1 sm:mt-0">
          Backend: {API_BASE_URL}
        </div>
      </header>

      <main className="flex-1 flex flex-col max-w-4xl mx-auto w-full p-4 gap-4">
        {/* Chat window */}
        <section className="flex-1 flex flex-col border border-slate-800 rounded-xl bg-slate-900/40">
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {messages.length === 0 && (
              <div className="text-xs text-slate-400 text-center mt-10">
                Ask anything about YantraLive rock breaker data, for example:
                <div className="mt-3 text-slate-300">
                  • “Which breaker works with SANY SY20?” <br />
                  • “What is the chisel diameter for VJ20 HD?” <br />
                  • “What is the price including GST for SKU00213749?” <br />
                  • “Is stock available for machine model TMX 20 Neo?”
                </div>
              </div>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${
                  m.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-xl px-3 py-2 text-sm whitespace-pre-wrap ${
                    m.role === "user"
                      ? "bg-sky-600 text-white"
                      : "bg-slate-800 text-slate-100"
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="text-xs text-slate-400 px-3 py-2 bg-slate-800 rounded-xl">
                  Thinking based only on your dataset…
                </div>
              </div>
            )}
          </div>

          {/* Input area */}
          <form
            className="border-t border-slate-800 p-2 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (!loading) handleSend();
            }}
          >
            <textarea
              className="flex-1 text-sm rounded-lg border border-slate-800 bg-slate-950/60 p-2 resize-none outline-none focus:ring-1 focus:ring-sky-500"
              rows={2}
              placeholder="Type your question about rock breakers, models, pricing, stock..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="text-sm px-4 py-2 rounded-lg border border-sky-500 bg-sky-600/80 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed transition self-end"
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}