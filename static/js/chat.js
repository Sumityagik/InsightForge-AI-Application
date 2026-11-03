/**
 * InsightForge AI — Chat interface logic.
 * File upload and image analysis are wired to the UI but the backend
 * endpoints for them don't exist yet (built in a later sprint) — the
 * attach button is disabled with a tooltip rather than faking the feature.
 */

let currentConversationId = null;
let allConversations = [];
let isSending = false;

// ============================================================
// INIT
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  loadCurrentUser();
  loadConversationHistory();
  bindEvents();
  autoResizeTextarea();
});

function bindEvents() {
  document.getElementById("new-chat-btn").addEventListener("click", createNewChat);
  document.getElementById("logout-btn").addEventListener("click", logout);
  document.getElementById("send-btn").addEventListener("click", sendMessage);
  document.getElementById("sidebar-toggle").addEventListener("click", toggleSidebar);
  document.getElementById("sidebar-close-btn").addEventListener("click", closeSidebar);
  document.getElementById("sidebar-overlay").addEventListener("click", closeSidebar);
  document.getElementById("search-conversations").addEventListener("input", handleSearch);
  document.getElementById("mic-btn").addEventListener("click", toggleVoiceRecognition);

  const input = document.getElementById("message-input");
  input.addEventListener("input", () => {
    document.getElementById("send-btn").disabled = input.value.trim().length === 0;
    autoResizeTextarea();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  document.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.suggestion;
      document.getElementById("send-btn").disabled = false;
      sendMessage();
    });
  });
}

function autoResizeTextarea() {
  const input = document.getElementById("message-input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

// ============================================================
// USER / AUTH
// ============================================================
async function loadCurrentUser() {
  try {
    const res = await fetch("/api/auth/me");
    const body = await res.json();
    if (body.status === "success") {
      const name = body.data.name || "User";
      document.getElementById("user-name-label").textContent = name;
      document.getElementById("user-avatar").textContent = name.charAt(0).toUpperCase();
      document.getElementById("current-title").textContent = "InsightForge AI";
    }
  } catch (e) {
    console.error("Failed to load current user", e);
  }
}

async function logout() {
  try {
    await fetch("/api/auth/logout", { method: "POST" });
  } finally {
    window.location.href = "/login";
  }
}

// ============================================================
// CONVERSATION HISTORY (sidebar)
// ============================================================
async function loadConversationHistory() {
  try {
    const res = await fetch("/api/chat/history");
    const body = await res.json();
    if (body.status !== "success") throw new Error(body.message);
    allConversations = body.data;
    renderConversationList(allConversations);
  } catch (e) {
    showToast("Could not load conversation history.", true);
  }
}

function renderConversationList(conversations) {
  const list = document.getElementById("conversation-list");
  list.innerHTML = "";

  if (conversations.length === 0) {
    list.innerHTML = '<div class="empty-history">No conversations yet</div>';
    return;
  }

  conversations.forEach((convo) => {
    const item = document.createElement("div");
    item.className = "conversation-item" + (convo.id === currentConversationId ? " active" : "");
    item.dataset.id = convo.id;

    const titleSpan = document.createElement("span");
    titleSpan.className = "title";
    titleSpan.textContent = convo.title || "New Chat";

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "delete-btn";
    deleteBtn.title = "Delete conversation";
    deleteBtn.innerHTML = "&#10005;";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(convo.id);
    });

    item.appendChild(titleSpan);
    item.appendChild(deleteBtn);
    item.addEventListener("click", () => selectConversation(convo.id));
    list.appendChild(item);
  });
}

function handleSearch(e) {
  const q = e.target.value.trim().toLowerCase();
  const filtered = q
    ? allConversations.filter((c) => (c.title || "").toLowerCase().includes(q))
    : allConversations;
  renderConversationList(filtered);
}

async function selectConversation(conversationId) {
  try {
    const res = await fetch(`/api/chat/${conversationId}/messages`);
    const body = await res.json();
    if (body.status !== "success") throw new Error(body.message);

    currentConversationId = conversationId;
    clearMessagesUI();
    showChatView();

    body.data.forEach((msg) => {
      if (msg.role === "user") addUserMessage(msg.content, false);
      else addAIMessage(msg.content, false);
    });

    scrollToBottom();
    renderConversationList(allConversations);
    closeSidebar();
  } catch (e) {
    showToast("Could not load that conversation.", true);
  }
}

async function deleteConversation(conversationId) {
  if (!confirm("Delete this conversation? This cannot be undone.")) return;
  try {
    const res = await fetch(`/api/chat/${conversationId}`, { method: "DELETE" });
    const body = await res.json();
    if (body.status !== "success") throw new Error(body.message);

    if (conversationId === currentConversationId) {
      createNewChatLocal();
    }
    await loadConversationHistory();
    showToast("Conversation deleted.");
  } catch (e) {
    showToast("Could not delete conversation.", true);
  }
}

function createNewChat() {
  createNewChatLocal();
}

function createNewChatLocal() {
  currentConversationId = null;
  clearChat();
}

function clearChat() {
  clearMessagesUI();
  showWelcomeView();
  renderConversationList(allConversations);
}

function clearMessagesUI() {
  document.getElementById("messages-inner").innerHTML = "";
}

function showChatView() {
  document.getElementById("welcome-screen").style.display = "none";
  document.getElementById("messages-inner").style.display = "block";
}

function showWelcomeView() {
  document.getElementById("welcome-screen").style.display = "flex";
  document.getElementById("messages-inner").style.display = "none";
}

// ============================================================
// SENDING MESSAGES
// ============================================================
async function sendMessage() {
  const input = document.getElementById("message-input");
  const content = input.value.trim();
  if (!content || isSending) return;

  isSending = true;
  document.getElementById("send-btn").disabled = true;

  showChatView();
  addUserMessage(content, true);
  input.value = "";
  autoResizeTextarea();
  showLoading();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: content,
        conversation_id: currentConversationId,
      }),
    });
    const body = await res.json();
    hideLoading();

    if (body.status !== "success") {
      showToast(body.message || "Something went wrong.", true);
      isSending = false;
      return;
    }

    currentConversationId = body.data.conversation.id;
    addAIMessage(body.data.ai_message.content, true);
    await loadConversationHistory();
  } catch (e) {
    hideLoading();
    showToast("Network error. Please check your connection and try again.", true);
  } finally {
    isSending = false;
    document.getElementById("send-btn").disabled = input.value.trim().length === 0;
  }
}

// ============================================================
// MESSAGE RENDERING
// ============================================================
function addUserMessage(content, scroll = true) {
  const el = buildMessageElement("user", content);
  document.getElementById("messages-inner").appendChild(el);
  if (scroll) scrollToBottom();
}

function addAIMessage(content, scroll = true) {
  const el = buildMessageElement("assistant", content);
  document.getElementById("messages-inner").appendChild(el);
  if (scroll) scrollToBottom();
}

function buildMessageElement(role, content) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "AI";

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "bubble-wrap";

  const label = document.createElement("div");
  label.className = "role-label";
  label.textContent = role === "user" ? "You" : "InsightForge AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(content);

  bubbleWrap.appendChild(label);
  bubbleWrap.appendChild(bubble);

  if (role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => copyToClipboard(content, copyBtn));
    actions.appendChild(copyBtn);

    bubbleWrap.appendChild(actions);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(bubbleWrap);
  return wrap;
}

function renderMarkdown(text) {
  if (window.marked) {
    return marked.parse(text);
  }
  // Fallback if the CDN script failed to load: escape HTML, preserve line breaks.
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return `<p>${escaped.replace(/\n/g, "<br>")}</p>`;
}

function copyToClipboard(text, button) {
  navigator.clipboard.writeText(text).then(() => {
    const original = button.textContent;
    button.textContent = "Copied!";
    setTimeout(() => (button.textContent = original), 1500);
  });
}

// ============================================================
// LOADING / TYPING INDICATOR
// ============================================================
function showLoading() {
  const el = document.createElement("div");
  el.className = "message assistant typing-indicator";
  el.id = "typing-indicator";
  el.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble-wrap">
      <div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
    </div>
  `;
  document.getElementById("messages-inner").appendChild(el);
  scrollToBottom();
}

function hideLoading() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function scrollToBottom() {
  const container = document.getElementById("messages-container");
  container.scrollTop = container.scrollHeight;
}

// ============================================================
// VOICE INPUT (Web Speech API)
// ============================================================
let recognition = null;
let isRecording = false;

function toggleVoiceRecognition() {
  if (isRecording) {
    stopVoiceRecognition();
  } else {
    startVoiceRecognition();
  }
}

function startVoiceRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const micBtn = document.getElementById("mic-btn");

  if (!SpeechRecognition) {
    showToast("Voice input isn't supported in this browser. Try Chrome or Edge.", true);
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.continuous = false;

  const input = document.getElementById("message-input");
  const baseText = input.value ? input.value + " " : "";

  recognition.onstart = () => {
    isRecording = true;
    micBtn.classList.add("recording");
    micBtn.title = "Listening... click to stop";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    input.value = baseText + transcript;
    document.getElementById("send-btn").disabled = input.value.trim().length === 0;
    autoResizeTextarea();
  };

  recognition.onerror = (event) => {
    if (event.error === "not-allowed" || event.error === "permission-denied") {
      showToast("Microphone permission was denied.", true);
    } else {
      showToast("Voice recognition error. Please try again.", true);
    }
    stopVoiceRecognition();
  };

  recognition.onend = () => {
    stopVoiceRecognition();
  };

  recognition.start();
}

function stopVoiceRecognition() {
  if (recognition) {
    recognition.stop();
    recognition = null;
  }
  isRecording = false;
  const micBtn = document.getElementById("mic-btn");
  micBtn.classList.remove("recording");
  micBtn.title = "Voice input";
}

// ============================================================
// SIDEBAR (mobile)
// ============================================================
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("sidebar-overlay").classList.toggle("visible");
}

function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("sidebar-overlay").classList.remove("visible");
}

// ============================================================
// TOASTS
// ============================================================
function showToast(message, isError = false) {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = "toast" + (isError ? " error" : "");
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}