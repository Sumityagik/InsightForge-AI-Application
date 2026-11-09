/**
 * InsightForge AI: document upload.
 *
 * Loaded AFTER chat.js. It reuses chat.js helpers (addUserMessage,
 * addAIMessage, showToast, loadConversationHistory ...) and shared state
 * (currentConversationId, isSending).
 *
 * Flow:
 *   paperclip -> pick file -> chip appears (name, type, size, remove)
 *   -> user presses Send (optionally with a question typed in the box)
 *   -> uploadDocument() sends the file with real progress
 *   -> "Analyzing <file>..." -> AI answer appears in the chat
 */

const SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "xlsx", "xls"];
const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"];
const MAX_FILE_MB = 20;

let pendingFile = null;

document.addEventListener("DOMContentLoaded", initUpload);

function initUpload() {
  const attachBtn = document.getElementById("attach-btn");
  const fileInput = document.getElementById("file-input");
  if (!attachBtn || !fileInput) return;

  attachBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
    fileInput.value = ""; // lets the user pick the same file again later
  });

  // chat.js disables Send when the text box is empty. With a file attached,
  // Send must stay enabled, so re-check after every keystroke.
  document.getElementById("message-input").addEventListener("input", refreshSendButton);
}

// ------------------------------------------------------------ helpers
function getExtension(name) {
  const i = name.lastIndexOf(".");
  return i === -1 ? "" : name.slice(i + 1).toLowerCase();
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function hasPendingFile() {
  return pendingFile !== null;
}

function refreshSendButton() {
  const text = document.getElementById("message-input").value.trim();
  document.getElementById("send-btn").disabled = isSending || (text.length === 0 && !pendingFile);
}

// ------------------------------------------------------------ choosing a file
function handleFileSelected(file) {
  const ext = getExtension(file.name);

  if (IMAGE_EXTENSIONS.includes(ext)) {
    showToast("Image upload isn't available yet. It will be added in a later update.", true);
    return;
  }
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    showToast("Unsupported file type. Please choose a PDF, DOCX, TXT, XLSX or XLS file.", true);
    return;
  }
  if (file.size === 0) {
    showToast("That file is empty.", true);
    return;
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    showToast(`That file is too large (max ${MAX_FILE_MB} MB).`, true);
    return;
  }

  pendingFile = file;
  renderFileChip();
  refreshSendButton();
  document.getElementById("message-input").focus();
}

function renderFileChip() {
  const row = document.getElementById("file-preview-row");
  row.innerHTML = "";
  if (!pendingFile) {
    row.classList.remove("visible");
    return;
  }

  const chip = document.createElement("div");
  chip.className = "file-chip";

  const info = document.createElement("div");
  info.className = "file-info";

  const name = document.createElement("span");
  name.className = "file-name";
  name.textContent = pendingFile.name; // textContent: no HTML injection
  name.title = pendingFile.name;

  const meta = document.createElement("span");
  meta.className = "file-meta";
  meta.id = "file-meta";
  meta.textContent = `${getExtension(pendingFile.name).toUpperCase()} · ${formatFileSize(pendingFile.size)}`;

  const progress = document.createElement("div");
  progress.className = "file-progress";
  progress.id = "file-progress";
  const bar = document.createElement("div");
  bar.className = "file-progress-bar";
  bar.id = "file-progress-bar";
  progress.appendChild(bar);

  info.appendChild(name);
  info.appendChild(meta);
  info.appendChild(progress);

  const remove = document.createElement("button");
  remove.className = "remove-file";
  remove.type = "button";
  remove.title = "Remove file";
  remove.innerHTML = "&#10005;";
  remove.addEventListener("click", removePendingFile);

  chip.appendChild(info);
  chip.appendChild(remove);
  row.appendChild(chip);
  row.classList.add("visible");
}

function removePendingFile() {
  if (isSending) return; // can't cancel mid-upload
  pendingFile = null;
  renderFileChip();
  refreshSendButton();
}

function setUploadProgress(percent) {
  const progress = document.getElementById("file-progress");
  const bar = document.getElementById("file-progress-bar");
  if (!progress || !bar) return;
  progress.classList.add("active");
  bar.style.width = percent + "%";
}

// ------------------------------------------------------------ status bubble
function showStatusBubble(text) {
  hideStatusBubble();
  const el = document.createElement("div");
  el.className = "message assistant typing-indicator";
  el.id = "status-bubble";
  el.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble-wrap">
      <div class="bubble">
        <span class="status-text"></span>
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    </div>`;
  el.querySelector(".status-text").textContent = text; // filename stays plain text
  document.getElementById("messages-inner").appendChild(el);
  scrollToBottom();
}

function updateStatusBubble(text) {
  const t = document.querySelector("#status-bubble .status-text");
  if (t) t.textContent = text;
}

function hideStatusBubble() {
  const el = document.getElementById("status-bubble");
  if (el) el.remove();
}

// ------------------------------------------------------------ upload
function postFileWithProgress(file, instruction, conversationId, onProgress, onSent) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.timeout = 180000; // analysis can take a while

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.upload.onload = onSent;

    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch (e) {
        /* non-JSON response, handled by caller */
      }
      resolve({ status: xhr.status, body });
    };
    xhr.onerror = () => reject(new Error("network"));
    xhr.ontimeout = () => reject(new Error("timeout"));

    const form = new FormData();
    form.append("file", file);
    if (instruction) form.append("message", instruction);
    if (conversationId) form.append("conversation_id", conversationId);
    xhr.send(form);
  });
}

async function sendWithFile() {
  if (isSending || !pendingFile) return;

  const file = pendingFile;
  const input = document.getElementById("message-input");
  const instruction = input.value.trim();

  isSending = true;
  document.getElementById("send-btn").disabled = true;

  showChatView();
  const label = `${getExtension(file.name).toUpperCase()}, ${formatFileSize(file.size)}`;
  // File name is user-controlled, so strip characters that could be read as HTML.
  const safeName = file.name.replace(/[<>&"]/g, "");
  addUserMessage(`**Uploaded:** ${safeName} (${label})` + (instruction ? `\n\n${instruction}` : ""), true);

  input.value = "";
  autoResizeTextarea();
  showStatusBubble(`Uploading ${file.name}...`);

  let keepFile = false;
  try {
    const { status, body } = await postFileWithProgress(
      file,
      instruction,
      currentConversationId,
      setUploadProgress,
      () => {
        setUploadProgress(100);
        updateStatusBubble(`Analyzing ${file.name}...`);
      }
    );

    hideStatusBubble();

    if (body && body.status === "success") {
      currentConversationId = body.data.conversation.id;
      addAIMessage(body.data.ai_message.content, true);
      await loadConversationHistory();
    } else {
      // The server may still have created the conversation (e.g. AI step failed).
      if (body && body.conversation_id) {
        currentConversationId = body.conversation_id;
        await loadConversationHistory();
      }
      showToast((body && body.message) || "The upload failed. Please try again.", true);
    }
  } catch (e) {
    hideStatusBubble();
    keepFile = true; // network problem: let the user retry with the same file
    showToast(
      e.message === "timeout"
        ? "The analysis took too long. Please try again."
        : "Network error. Please check your connection and try again.",
      true
    );
  } finally {
    isSending = false;
    if (!keepFile) pendingFile = null;
    renderFileChip();
    refreshSendButton();
  }
}