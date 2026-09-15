(() => {
  "use strict";

  const chatEl = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const showInternals = document.getElementById("show-internals");
  const assistantTpl = document.getElementById("tpl-assistant-message");

  const sessionId = getOrCreateSessionId();

  addSystemNote(
    "Ask about billing, the API, an outage, security, onboarding, or refunds. " +
      "Answers are grounded in the Nimbus Cloud knowledge base via RAG."
  );

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    setBusy(true);

    addUserMessage(message);
    const assistantNode = addEmptyAssistantMessage();

    try {
      await streamChat(message, assistantNode);
    } catch (err) {
      setAnswerText(assistantNode, `Sorry, something went wrong talking to the backend: ${err.message}`);
    } finally {
      setBusy(false);
    }
  });

  showInternals.addEventListener("change", () => {
    document.querySelectorAll(".internals").forEach((el) => {
      el.hidden = !showInternals.checked;
    });
  });

  async function streamChat(message, node) {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const answerEl = node.querySelector(".answer");
    answerEl.classList.add("typing");

    let interactionId = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseSseEvent(rawEvent);
        if (!parsed) continue;

        if (parsed.event === "meta") {
          const meta = JSON.parse(parsed.data);
          interactionId = meta.interaction_id;
          applyMeta(node, meta);
        } else if (parsed.event === "token") {
          const { text } = JSON.parse(parsed.data);
          answerEl.textContent += text;
          chatEl.scrollTop = chatEl.scrollHeight;
        } else if (parsed.event === "done") {
          const { latency_ms } = JSON.parse(parsed.data);
          setLatency(node, latency_ms);
        }
      }
    }

    answerEl.classList.remove("typing");
    wireFeedback(node, interactionId);
  }

  function parseSseEvent(raw) {
    let event = "message";
    let data = "";
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) return null;
    return { event, data };
  }

  function applyMeta(node, meta) {
    node.querySelector(".intent").textContent = `intent: ${meta.intent}`;
    node.querySelector(".arm").textContent = `retrieval strategy: ${meta.arm}`;

    const list = node.querySelector(".source-list");
    list.innerHTML = "";
    if (meta.sources.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No matching knowledge base passages found.";
      list.appendChild(li);
    }
    for (const s of meta.sources) {
      const li = document.createElement("li");
      const scorePct = Math.round(s.score * 100);
      li.innerHTML = `<strong>${escapeHtml(s.source)}</strong> <span class="source-score">(${scorePct}% match)</span>`;
      list.appendChild(li);
    }
  }

  function setLatency(node, latencyMs) {
    node.querySelector(".latency").textContent = `${Math.round(latencyMs)} ms`;
  }

  function wireFeedback(node, interactionId) {
    const upBtn = node.querySelector(".fb-up");
    const downBtn = node.querySelector(".fb-down");
    const ack = node.querySelector(".fb-ack");

    if (!interactionId) {
      upBtn.disabled = true;
      downBtn.disabled = true;
      return;
    }

    const send = async (rating) => {
      upBtn.disabled = true;
      downBtn.disabled = true;
      try {
        const resp = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ interaction_id: interactionId, rating }),
        });
        if (resp.ok) {
          (rating === "up" ? upBtn : downBtn).classList.add(
            rating === "up" ? "selected-up" : "selected-down"
          );
          ack.hidden = false;
        }
      } catch (_) {
        upBtn.disabled = false;
        downBtn.disabled = false;
      }
    };

    upBtn.addEventListener("click", () => send("up"));
    downBtn.addEventListener("click", () => send("down"));
  }

  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "message user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    chatEl.appendChild(wrap);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function addEmptyAssistantMessage() {
    const fragment = assistantTpl.content.cloneNode(true);
    const node = fragment.querySelector(".message");
    chatEl.appendChild(node);
    node.querySelector(".internals").hidden = !showInternals.checked;
    chatEl.scrollTop = chatEl.scrollHeight;
    return node;
  }

  function setAnswerText(node, text) {
    node.querySelector(".answer").textContent = text;
    node.querySelector(".answer").classList.remove("typing");
  }

  function addSystemNote(text) {
    const note = document.createElement("div");
    note.className = "system-note";
    note.textContent = text;
    chatEl.appendChild(note);
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    input.disabled = busy;
  }

  function getOrCreateSessionId() {
    const key = "nimbus_copilot_session_id";
    let id = localStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(key, id);
    }
    return id;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
