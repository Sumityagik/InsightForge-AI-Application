/**
 * Auth form handling for InsightForge AI.
 * Shared by login.html and register.html.
 */

function showAuthError(message) {
  const el = document.getElementById("auth-error");
  if (!el) return;
  el.textContent = message;
  el.classList.add("visible");
}

function hideAuthError() {
  const el = document.getElementById("auth-error");
  if (!el) return;
  el.classList.remove("visible");
  el.textContent = "";
}

function setSubmitting(button, isSubmitting, idleText) {
  button.disabled = isSubmitting;
  button.textContent = isSubmitting ? "Please wait..." : idleText;
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let body;
  try {
    body = await response.json();
  } catch (e) {
    throw new Error("Unexpected server response. Please try again.");
  }
  if (!response.ok || body.status !== "success") {
    throw new Error(body.message || "Something went wrong.");
  }
  return body;
}

function initLoginForm() {
  const form = document.getElementById("login-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAuthError();

    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const remember = document.getElementById("remember")
      ? document.getElementById("remember").checked
      : false;
    const button = form.querySelector(".auth-submit");

    if (!email || !password) {
      showAuthError("Please enter both email and password.");
      return;
    }

    setSubmitting(button, true, "Log In");
    try {
      await postJSON("/api/auth/login", { email, password, remember });
      window.location.href = "/chat";
    } catch (err) {
      showAuthError(err.message);
      setSubmitting(button, false, "Log In");
    }
  });
}

function initRegisterForm() {
  const form = document.getElementById("register-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAuthError();

    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirm_password").value;
    const button = form.querySelector(".auth-submit");

    if (!name || !email || !password || !confirmPassword) {
      showAuthError("Please fill in all fields.");
      return;
    }
    if (password !== confirmPassword) {
      showAuthError("Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      showAuthError("Password must be at least 8 characters long.");
      return;
    }

    setSubmitting(button, true, "Create Account");
    try {
      await postJSON("/api/auth/register", {
        name,
        email,
        password,
        confirm_password: confirmPassword,
      });
      window.location.href = "/chat";
    } catch (err) {
      showAuthError(err.message);
      setSubmitting(button, false, "Create Account");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initLoginForm();
  initRegisterForm();
});
