/* SIFT waitlist form — progressive enhancement.
   Submits to POST /api/waitlist (see web/api/waitlist.py for the local stub).
   With JS disabled the form still POSTs (plain HTML form), so the page works
   on any static host plus the tiny endpoint, or any host that maps /api/waitlist. */
(function () {
  "use strict";

  var form = document.getElementById("waitlist-form");
  var msg = document.getElementById("form-msg");
  var input = document.getElementById("email");
  var tos = document.getElementById("tos-accepted");
  var button = form.querySelector("button[type=submit]");

  var EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

  function setMsg(text, kind) {
    msg.textContent = text;
    msg.className = "form-msg" + (kind ? " " + kind : "");
  }

  // Onboarding copy (first-run): fetched from GET /api/onboarding when the
  // endpoint is up; the static HTML already carries the same statement, so
  // the page is correct with JS disabled or on a pure static host.
  fetch("/api/onboarding", { headers: { "Accept": "application/json" } })
    .then(function (res) { return res.ok ? res.json() : null; })
    .then(function (body) {
      if (body && body.ok && body.onboarding) {
        var el = document.getElementById("onboarding-statement");
        if (el && body.onboarding.statement) { el.textContent = body.onboarding.statement; }
      }
    })
    .catch(function () { /* static copy already in the HTML — nothing to do */ });

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var email = input.value.trim();

    if (!EMAIL_RE.test(email)) {
      setMsg("Please enter a valid email address.", "err");
      input.focus();
      return;
    }
    // Day-one ToS acceptance is mandatory (fail closed server-side too).
    if (!tos || !tos.checked) {
      setMsg("Please accept the terms to join the waitlist.", "err");
      if (tos) { tos.focus(); }
      return;
    }

    button.disabled = true;
    setMsg("Sending…", "");

    fetch(form.action, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ email: email, tos_accepted: tos ? tos.checked : false })
    })
      .then(function (res) {
        return res.json().then(function (body) {
          return { ok: res.ok, body: body };
        });
      })
      .then(function (r) {
        if (r.ok && r.body && r.body.ok) {
          setMsg("You're on the list — invite waves go out as beta capacity opens.", "ok");
          form.reset();
        } else {
          setMsg((r.body && r.body.error) || "Something went wrong. Please try again.", "err");
        }
      })
      .catch(function () {
        // Endpoint stub may not be running on a pure static host — fall back to a
        // plain form POST so the email still reaches the server-side store.
        var fd = new FormData();
        fd.append("email", email);
        if (tos) { fd.append("tos_accepted", tos.checked ? "yes" : ""); }
        fetch(form.action, { method: "POST", body: fd })
          .then(function (res) {
            if (res.ok) {
              setMsg("You're on the list — invite waves go out as beta capacity opens.", "ok");
              form.reset();
            } else {
              setMsg("Could not reach the waitlist endpoint. Email-only sign-ups need the endpoint running — see README.", "err");
            }
          })
          .catch(function () {
            setMsg("Could not reach the waitlist endpoint. Email-only sign-ups need the endpoint running — see README.", "err");
          });
      })
      .finally(function () {
        button.disabled = false;
      });
  });
})();
