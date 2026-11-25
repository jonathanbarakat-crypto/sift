/* SIFT Gate-1 Invited Beta UI controller.
   All card, onboarding, ToS, glossary, gauge, health, and verdict copy is
   rendered strictly from server responses built in sift/cards/copy.py +
   sift/decks/copy.py + sift/cards/face.py. Zero raw market data or tickers. */
(function () {
  "use strict";

  var state = {
    user: null,
    onboarding: null,
    catalog: [],
    subspaces: [],
    warehouseReady: false,
    activeSource: null,
    pollTimer: null,
    selectedCardId: null
  };

  function $(id) { return document.getElementById(id); }

  function setMsg(el, text, kind) {
    if (!el) return;
    el.textContent = text || "";
    el.className = "form-msg" + (kind ? " " + kind : "");
  }

  function api(url, opts) {
    var cfg = Object.assign({ credentials: "same-origin", headers: {} }, opts || {});
    if (cfg.body && typeof cfg.body === "object") {
      cfg.headers["Content-Type"] = "application/json";
      cfg.body = JSON.stringify(cfg.body);
    }
    cfg.headers["Accept"] = "application/json";
    return fetch(url, cfg).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        return { ok: res.ok, status: res.status, data: data };
      });
    });
  }

  function tierTagClass(tier) {
    if (tier === "platinum") return "p";
    if (tier === "gold") return "g";
    if (tier === "silver") return "s";
    return "b";
  }

  function clearEl(el) {
    while (el && el.firstChild) { el.removeChild(el.firstChild); }
  }

  function mk(tag, attrs, text) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "className") el.className = attrs[k];
        else el.setAttribute(k, attrs[k]);
      });
    }
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }

  function clearAllPrivatePanels() {
    clearEl($("live-private-spec-panel"));
    clearEl($("trial-private-spec-panel"));
    var cp = $("card-private-spec-slot");
    if (cp) clearEl(cp);
  }

  // ---- onboarding copy (server-rendered only) ----
  function loadOnboarding() {
    return api("/api/onboarding").then(function (r) {
      if (!r.ok || !r.data || !r.data.onboarding) return;
      var ob = r.data.onboarding;
      state.onboarding = ob;
      $("auth-onboarding-statement").textContent = ob.statement || "";
      $("auth-tos-text").textContent = ob.tos_acceptance || "";
      $("auth-candidates-line").textContent = ob.cards_are_candidates || "";
      $("auth-privacy-note").textContent = ob.privacy_note || "";
      $("mine-candidates-line").textContent = ob.cards_are_candidates || "";
      $("vault-hit-hint").textContent = ob.hit_loop_hint || "";
      $("footer-statement").textContent = ob.statement || "";
    });
  }

  // ---- auth view <-> app view ----
  function showApp(user) {
    state.user = user;
    $("auth-view").classList.add("hidden");
    $("app-view").classList.remove("hidden");
    $("logout-btn").classList.remove("hidden");
    $("user-id-display").textContent = user.user_id;
    $("kill-rate-toggle").checked = !!user.kill_rate_opt_in;
    loadCatalog();
    loadTrials();
    loadVault();
  }

  function showAuth() {
    state.user = null;
    state.selectedCardId = null;
    stopStream();
    clearAllPrivatePanels();
    $("app-view").classList.add("hidden");
    $("auth-view").classList.remove("hidden");
    $("logout-btn").classList.add("hidden");
  }

  function checkSession() {
    return api("/api/auth/me").then(function (r) {
      if (r.ok && r.data && r.data.user) showApp(r.data.user);
      else showAuth();
    });
  }

  // ---- tab switching ----
  document.querySelectorAll("[data-auth-tab]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var tab = btn.getAttribute("data-auth-tab");
      document.querySelectorAll("[data-auth-tab]").forEach(function (b) {
        b.classList.toggle("active", b === btn);
      });
      $("signup-form").classList.toggle("hidden", tab !== "signup");
      $("login-form").classList.toggle("hidden", tab !== "login");
    });
  });

  document.querySelectorAll("[data-app-tab]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var tab = btn.getAttribute("data-app-tab");
      document.querySelectorAll("[data-app-tab]").forEach(function (b) {
        b.classList.toggle("active", b === btn);
      });
      clearAllPrivatePanels();
      $("tab-mine").classList.toggle("hidden", tab !== "mine");
      $("tab-trials").classList.toggle("hidden", tab !== "trials");
      $("tab-vault").classList.toggle("hidden", tab !== "vault");
      if (tab === "trials") loadTrials();
      if (tab === "vault") loadVault();
    });
  });

  // ---- signup / login / logout / profile ----
  $("signup-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var msg = $("su-msg");
    if (!$("su-tos").checked) {
      setMsg(msg, "Please accept the day-one terms to create your beta account.", "err");
      return;
    }
    $("su-submit").disabled = true;
    setMsg(msg, "Creating beta account…", "");
    api("/api/auth/signup", {
      method: "POST",
      body: {
        invite_code: $("su-invite").value.trim(),
        email: $("su-email").value.trim(),
        password: $("su-password").value,
        tos_accepted: $("su-tos").checked
      }
    }).then(function (r) {
      if (r.ok && r.data && r.data.user) {
        $("signup-form").reset();
        setMsg(msg, "", "");
        showApp(r.data.user);
      } else {
        setMsg(msg, (r.data && r.data.error) || "Signup failed.", "err");
      }
    }).finally(function () {
      $("su-submit").disabled = false;
    });
  });

  $("login-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var msg = $("li-msg");
    $("li-submit").disabled = true;
    setMsg(msg, "Signing in…", "");
    api("/api/auth/login", {
      method: "POST",
      body: {
        email: $("li-email").value.trim(),
        password: $("li-password").value
      }
    }).then(function (r) {
      if (r.ok && r.data && r.data.user) {
        $("login-form").reset();
        setMsg(msg, "", "");
        showApp(r.data.user);
      } else {
        setMsg(msg, (r.data && r.data.error) || "Sign-in failed.", "err");
      }
    }).finally(function () {
      $("li-submit").disabled = false;
    });
  });

  $("logout-btn").addEventListener("click", function () {
    api("/api/auth/logout", { method: "POST" }).then(function () { showAuth(); });
  });

  $("kill-rate-toggle").addEventListener("change", function () {
    var enabled = $("kill-rate-toggle").checked;
    api("/api/auth/profile", {
      method: "PATCH",
      body: { kill_rate_opt_in: enabled }
    }).then(function (r) {
      if (r.ok && r.data && r.data.user) {
        state.user = r.data.user;
        loadTrials();
      }
    });
  });

  // ---- catalog, unified Slot A / Slot B builder & mining ----
  function fillSelect(el, values, defVal) {
    clearEl(el);
    values.forEach(function (v) {
      var opt = mk("option", { value: String(v) }, String(v));
      if (String(v) === String(defVal)) opt.selected = true;
      el.appendChild(opt);
    });
  }

  function populateSlotSelect(selectEl) {
    var prev = selectEl.value;
    clearEl(selectEl);
    var subs = state.subspaces.length
      ? state.subspaces
      : [{ id: "all", label: "Random (All eligible)", eligible_count: 0 }];
    subs.forEach(function (s) {
      var lbl = s.eligible_count ? (s.label + " · " + s.eligible_count) : s.label;
      selectEl.appendChild(mk("option", { value: "random:" + s.id }, lbl));
    });
    selectEl.appendChild(mk("option", { value: "symbol" }, "Specific stock…"));
    if (prev && Array.prototype.some.call(selectEl.options, function (o) { return o.value === prev; })) {
      selectEl.value = prev;
    } else {
      selectEl.value = "random:all";
    }
  }

  function subspaceById(id) {
    return state.subspaces.find(function (s) { return s.id === id; }) || null;
  }

  function readSlot(modeId, symId) {
    var v = $(modeId).value || "random:all";
    if (v === "symbol") {
      var raw = ($(symId).value || "").trim().toUpperCase();
      return { kind: "symbol", value: raw };
    }
    var sub = v.indexOf("random:") === 0 ? v.slice(7) : "all";
    return { kind: "random", subspace: sub };
  }

  function currentFamily() {
    var sel = $("family-select");
    return (sel && sel.value === "lead_lag") ? "lead_lag" : "pairs";
  }

  function applyFamilyControls() {
    var fam = currentFamily();
    var isLL = fam === "lead_lag";
    if ($("slot-a-label")) $("slot-a-label").textContent = isLL ? "Leader (Slot A)" : "Slot A";
    if ($("slot-b-label")) $("slot-b-label").textContent = isLL ? "Follower (Slot B)" : "Slot B";
    if ($("closez-wrap")) $("closez-wrap").classList.toggle("hidden", isLL);
    if ($("lag-wrap")) $("lag-wrap").classList.toggle("hidden", !isLL);

    var cfgData = state.familyConfigs || {};
    if (isLL) {
      var ll = cfgData.lead_lag || {};
      fillSelect($("form-select"), ll.allowed_formations || [60, 120, 252], 120);
      fillSelect($("openz-select"), ll.allowed_open_z || [1.0, 1.5, 2.0], 1.5);
      fillSelect($("lag-select"), ll.allowed_lags || [1, 2, 5], 1);
    } else {
      var pr = cfgData.pairs || {};
      fillSelect($("form-select"), pr.allowed_formations || [40, 60, 80], 60);
      fillSelect($("openz-select"), pr.allowed_open_z || [1.5, 2.0, 2.5], 2.0);
      fillSelect($("closez-select"), pr.allowed_close_z || [0.3, 0.5, 0.8], 0.5);
    }
    updateSlotUI();
  }

  function updateSlotUI() {
    var isLL = currentFamily() === "lead_lag";
    var arrow = isLL ? "→" : "↔";
    var aIsSym = $("slot-a-mode").value === "symbol";
    var bIsSym = $("slot-b-mode").value === "symbol";
    $("slot-a-symbol").classList.toggle("hidden", !aIsSym);
    $("slot-b-symbol").classList.toggle("hidden", !bIsSym);

    var pill = $("slot-mode-pill");
    var meta = $("slot-subspace-meta");
    var summary = $("slot-mode-summary");

    if (!state.warehouseReady) {
      pill.className = "pill warn";
      pill.textContent = "Synthetic onboarding mode";
      meta.textContent = "PIT warehouse not mounted";
      summary.textContent = "Use the guided synthetic example below to walk through S0–S4 and automatic S5 validation.";
      $("run-gauntlet-btn").disabled = true;
      return;
    }
    $("run-gauntlet-btn").disabled = false;

    var sa = readSlot("slot-a-mode", "slot-a-symbol");
    var sb = readSlot("slot-b-mode", "slot-b-symbol");
    if (sa.kind === "symbol" && sb.kind === "symbol") {
      pill.className = "pill ok";
      pill.textContent = "Direct Hypothesis (A " + arrow + " B)";
      meta.textContent = isLL ? "Directed: Leader A → Follower B" : "Subspace: 1 exact hypothesis";
      summary.textContent = isLL
        ? "Tests your directed Leader → Follower hypothesis with cross-sectional demeaning, AR(1) pre-whitening, and HAC standard errors. S5 full-depth check runs automatically."
        : "Tests your exact two-stock hypothesis directly on PIT history. S5 full-depth check runs automatically.";
    } else if (sa.kind === "symbol" || sb.kind === "symbol") {
      var subId = sa.kind === "random" ? sa.subspace : sb.subspace;
      var subInfo = subspaceById(subId);
      var cnt = subInfo ? Math.max(subInfo.eligible_count - 1, 0) : "?";
      pill.className = "pill inf";
      var anchorLbl = isLL
        ? (sa.kind === "symbol" ? "Anchored Leader (A → ?)" : "Anchored Follower (? → B)")
        : "Anchored Search (A ↔ ?)";
      pill.textContent = anchorLbl;
      meta.textContent = "Anchor locked · " + (subInfo ? subInfo.label : subId) + " (~" + cnt + " partners)";
      summary.textContent = "Locks your anchor and draws a partner uniformly via PCG64 with zero hidden pre-screening. S5 full-depth check runs automatically.";
    } else {
      var infoA = subspaceById(sa.subspace);
      var infoB = subspaceById(sb.subspace);
      var pairEstimate = "?";
      if (infoA && infoB) {
        if (sa.subspace === sb.subspace) {
          pairEstimate = isLL
            ? String(infoA.eligible_count * (infoA.eligible_count - 1))
            : String(Math.floor(infoA.eligible_count * (infoA.eligible_count - 1) / 2));
        } else {
          pairEstimate = String(infoA.eligible_count * infoB.eligible_count);
        }
      }
      pill.className = "pill warn";
      pill.textContent = "Uniform Random (? " + arrow + " ?)";
      meta.textContent = "Zero pre-screen · ~" + pairEstimate + (isLL ? " directed pairs" : " eligible pairs");
      summary.textContent = "Draws uniformly via PCG64 across your chosen PIT subspace with zero data-peeking pre-screen (low hit rate on pure noise is expected). S5 full-depth check runs automatically.";
    }
  }

  function loadCatalog() {
    api("/api/mining/catalog").then(function (r) {
      if (!r.ok || !r.data) return;
      state.catalog = r.data.candidates || [];
      state.subspaces = r.data.subspaces || [];
      state.warehouseReady = !!r.data.warehouse_ready;
      state.familyConfigs = {
        pairs: {
          allowed_formations: r.data.allowed_formations || [40, 60, 80],
          allowed_open_z: r.data.allowed_open_z || [1.5, 2.0, 2.5],
          allowed_close_z: r.data.allowed_close_z || [0.3, 0.5, 0.8]
        },
        lead_lag: r.data.lead_lag_config || {
          allowed_lags: [1, 2, 5],
          allowed_formations: [60, 120, 252],
          allowed_open_z: [1.0, 1.5, 2.0]
        }
      };

      var famSel = $("family-select");
      if (famSel) {
        var allowedFams = r.data.allowed_families || ["pairs"];
        var prevFam = famSel.value;
        clearEl(famSel);
        allowedFams.forEach(function (f) {
          var label = f === "lead_lag"
            ? "Lead–Lag (Leader A → Follower B · directed)"
            : "Pairs (A ↔ B · mean-reversion)";
          famSel.appendChild(mk("option", { value: f }, label));
        });
        if (prevFam && allowedFams.indexOf(prevFam) !== -1) {
          famSel.value = prevFam;
        }
      }

      populateSlotSelect($("slot-a-mode"));
      populateSlotSelect($("slot-b-mode"));
      fillSelect($("scope-select"), r.data.allowed_scopes || ["free_5y_top1k"], "free_5y_top1k");

      var hasExample = state.catalog.some(function (c) { return c.synthetic_example; });
      $("run-example-btn").classList.toggle("hidden", !hasExample);
      applyFamilyControls();
    });
  }

  if ($("family-select")) {
    $("family-select").addEventListener("change", applyFamilyControls);
  }
  ["slot-a-mode", "slot-b-mode"].forEach(function (id) {
    $(id).addEventListener("change", updateSlotUI);
  });
  ["slot-a-symbol", "slot-b-symbol"].forEach(function (id) {
    $(id).addEventListener("blur", function () {
      $(id).value = ($(id).value || "").trim().toUpperCase();
    });
  });

  // ---- Private Owner-Only Spec renderer (strictly separate from shareable card face) ----
  function renderPrivateSpecInto(container, url) {
    if (!container) return;
    clearEl(container);
    var loading = mk("div", { className: "private-spec-box" });
    loading.appendChild(mk("p", { className: "fine-print mono", style: "margin:0" }, "Loading authenticated owner-only private spec…"));
    container.appendChild(loading);

    api(url).then(function (r) {
      clearEl(container);
      var box = mk("div", { className: "private-spec-box" });
      if (!r.ok || !r.data || !r.data.private_spec) {
        box.appendChild(mk("p", { className: "form-msg err", style: "margin:0" }, (r.data && r.data.error) || "Private spec unavailable."));
        container.appendChild(box);
        return;
      }
      var ps = r.data.private_spec;
      var p = ps.parameters || {};
      var isLL = p.family === "lead_lag";
      var banner = mk("div", { className: "private-banner" });
      banner.appendChild(mk("span", { className: "pill warn" }, "PRIVATE · OWNER ONLY · DO NOT SHARE"));
      var closeBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Hide");
      closeBtn.addEventListener("click", function () { clearEl(container); });
      banner.appendChild(closeBtn);
      box.appendChild(banner);

      // Instrument identifiers + copy helper
      var insts = ps.instruments || [];
      if (insts.length) {
        var symList = insts.map(function (it) { return it.symbol; });
        var chips = mk("div", { className: "instrument-chips" });
        insts.forEach(function (it, idx) {
          if (idx > 0) chips.appendChild(mk("span", { className: "mono", style: "color:var(--ink-dim)" }, isLL ? "→" : "↔"));
          var lbl = isLL ? (it.role + ": " + it.symbol) : it.symbol;
          chips.appendChild(mk("span", { className: "instrument-chip" }, lbl));
        });
        var copyBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Copy symbols");
        copyBtn.addEventListener("click", function () {
          var txt = symList.join(isLL ? " -> " : " / ");
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(txt).then(function () {
              copyBtn.textContent = "Copied (" + txt + ")";
            });
          } else {
            copyBtn.textContent = txt;
          }
        });
        chips.appendChild(copyBtn);
        box.appendChild(chips);
      } else {
        box.appendChild(mk("p", { className: "fine-print mono" }, "Synthetic guided example — no live equity symbols."));
      }

      // Parameters (family, mode, subspace, formation, thresholds — zero prices/slopes)
      var paramSummary =
        "family=" + (p.family || "pairs") +
        " · mode=" + (p.mining_mode || "direct") +
        " · subspace=" + (p.subspace_filter || "all") +
        " (N=" + (p.subspace_size || 1) + ")" +
        " · scope=" + (p.scope_id || "") +
        " · formation=" + (p.formation || 60) +
        " · entry |z|=" + (p.open_z || 2.0) +
        (isLL
          ? (" · lag/hold=" + (p.lag_bars || 1) + "d")
          : (" · exit |z|=" + (p.close_z || 0.5)));
      box.appendChild(mk("p", { className: "mono fine-print", style: "margin:6px 0" }, paramSummary));

      // Live magnitude-only signal
      var ls = ps.live_signal || {};
      var zTxt = (ls.magnitude_z !== null && ls.magnitude_z !== undefined) ? ("|z| = " + ls.magnitude_z) : "|z| unavailable";
      box.appendChild(mk(
        "p",
        { className: "mono", style: "margin:8px 0 4px;font-size:0.86rem;color:var(--ink)" },
        "Live magnitude: " + zTxt + " (" + (ls.band || "normal") + " — " + (ls.label || "") + ")"
      ));

      // Durability & health
      var dh = ps.durability_and_health || {};
      box.appendChild(mk(
        "p",
        { className: "fine-print", style: "margin:4px 0" },
        "Health: " + (dh.health_phrase || "holding near its mint baseline") +
        (dh.hits_trailing_12m !== null && dh.hits_trailing_12m !== undefined ? (" · trailing-12m hits: " + dh.hits_trailing_12m) : "") +
        (dh.half_life_days !== null && dh.half_life_days !== undefined ? (" · half-life: " + dh.half_life_days + "d") : "")
      ));

      if (ps.guidance) {
        box.appendChild(mk("p", { className: "serif", style: "margin:10px 0 6px;font-size:0.9rem;color:var(--ink)" }, ps.guidance));
      }
      box.appendChild(mk("p", { className: "fine-print", style: "margin:6px 0 0" }, ps.disclaimer || ""));
      container.appendChild(box);
    });
  }

  function stopStream() {
    if (state.activeSource) {
      try { state.activeSource.close(); } catch (_) {}
      state.activeSource = null;
    }
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function resetCeremony() {
    ["seg-flicker", "seg-amber", "seg-s5"].forEach(function (id) {
      $(id).className = "ceremony-seg";
    });
    renderStages([]);
    $("live-verdict").classList.add("hidden");
    clearEl($("live-private-spec-panel"));
  }

  function renderStages(stageList) {
    var map = {};
    (stageList || []).forEach(function (s) { map[s.stage] = s; });
    var order = ["s0", "s1", "s2", "s4", "s3", "s5"];
    var wrap = $("live-stages");
    clearEl(wrap);
    order.forEach(function (st) {
      var row = map[st];
      var status = row ? row.status : "pending";
      var label = st.toUpperCase() + " " + status;
      if (row && row.p_value !== null && row.p_value !== undefined && st === "s2") {
        label += " (p=" + row.p_value + ")";
      } else if (row && row.dsr !== null && row.dsr !== undefined && st === "s3") {
        label += " (DSR=" + row.dsr + ")";
      }
      wrap.appendChild(mk("span", { className: "stage " + status }, label));
    });
    if (map.s0 || map.s1 || map.s2) {
      var killedEarly = (map.s0 && map.s0.status === "fail") ||
                        (map.s1 && map.s1.status === "fail") ||
                        (map.s2 && map.s2.status === "fail");
      $("seg-flicker").className = "ceremony-seg " + (killedEarly ? "seg-killed" : "active-flicker");
    }
    if (map.s4 || map.s3) {
      $("seg-amber").className = "ceremony-seg active-amber";
    }
    if (map.s5 && map.s5.status !== "pending") {
      $("seg-s5").className = "ceremony-seg " + (map.s5.status === "pass" ? "active-s5" : "active-amber");
    }
  }

  function renderVerdict(v) {
    var box = $("live-verdict");
    box.classList.remove("hidden", "killed", "amber", "green");
    var pill = $("live-verdict-pill");
    if (v.verdict === "killed") {
      box.classList.add("killed");
      pill.className = "pill err";
      pill.textContent = "KILLED @ " + String(v.killed_at || "screen").toUpperCase();
    } else if (v.survivor_class === "validated-green") {
      box.classList.add("green");
      pill.className = "pill ok";
      pill.textContent = "VALIDATED-GREEN · SURVIVOR";
    } else {
      box.classList.add("amber");
      pill.className = "pill warn";
      pill.textContent = String(v.survivor_class || "on-watch-amber").toUpperCase();
    }
    $("live-trial-id").textContent = v.trial_id + (v.mining_mode ? (" · " + v.mining_mode) : "") + (v.cached ? " · skip-list cached" : "");
    $("live-verdict-copy").textContent = v.verdict_copy || "";

    var actions = $("live-verdict-actions");
    clearEl(actions);
    if (v.card_id) {
      var inspectBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Inspect minted card");
      inspectBtn.addEventListener("click", function () {
        document.querySelector('[data-app-tab="vault"]').click();
        inspectCard(v.card_id);
      });
      var saveBtn = mk("button", { type: "button", className: "btn btn-brass btn-sm" }, "Save connection");
      saveBtn.addEventListener("click", function () {
        saveConnection(v.card_id);
      });
      actions.appendChild(inspectBtn);
      actions.appendChild(saveBtn);
    }
    if (v.private_spec_eligible && (v.card_id || v.trial_id)) {
      var privUrl = v.card_id
        ? ("/api/private/cards/" + encodeURIComponent(v.card_id) + "/spec")
        : ("/api/private/trials/" + encodeURIComponent(v.trial_id) + "/spec");
      var privBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Reveal private spec");
      privBtn.addEventListener("click", function () {
        renderPrivateSpecInto($("live-private-spec-panel"), privUrl);
      });
      actions.appendChild(privBtn);
    }
  }

  function startMining(isExample) {
    stopStream();
    resetCeremony();
    var msg = $("mine-msg");
    var fam = currentFamily();
    var isLL = fam === "lead_lag";
    var payload;

    if (isExample) {
      if (isLL) {
        payload = {
          family: "lead_lag",
          example: true,
          scope_id: "free_5y_top1k",
          lag_bars: 1,
          formation: 120,
          open_z: 1.5
        };
      } else {
        payload = {
          family: "pairs",
          example: true,
          scope_id: "free_5y_top1k",
          formation: 60,
          open_z: 2.0,
          close_z: 0.5
        };
      }
    } else {
      var sa = readSlot("slot-a-mode", "slot-a-symbol");
      var sb = readSlot("slot-b-mode", "slot-b-symbol");
      if (sa.kind === "symbol" && !sa.value) {
        setMsg(msg, "Enter a ticker symbol in " + (isLL ? "Leader (Slot A)" : "Slot A") + " or choose a Random subspace.", "err");
        return;
      }
      if (sb.kind === "symbol" && !sb.value) {
        setMsg(msg, "Enter a ticker symbol in " + (isLL ? "Follower (Slot B)" : "Slot B") + " or choose a Random subspace.", "err");
        return;
      }
      payload = {
        family: fam,
        slot_a: sa,
        slot_b: sb,
        scope_id: $("scope-select").value,
        formation: parseInt($("form-select").value, 10),
        open_z: parseFloat($("openz-select").value)
      };
      if (isLL) {
        payload.lag_bars = parseInt($("lag-select").value, 10);
      } else {
        payload.close_z = parseFloat($("closez-select").value);
      }
    }

    setMsg(msg, "Pre-registering trial in ledger…", "");
    $("run-gauntlet-btn").disabled = true;
    $("run-example-btn").disabled = true;

    api("/api/mining/runs", { method: "POST", body: payload }).then(function (r) {
      if (!r.ok || !r.data || !r.data.run) {
        setMsg(msg, (r.data && r.data.error) || "Could not start mining run.", "err");
        $("run-gauntlet-btn").disabled = !state.warehouseReady;
        $("run-example-btn").disabled = false;
        return;
      }
      var run = r.data.run;
      $("run-meta-line").textContent =
        "run=" + run.run_id +
        " · trial=" + run.trial_id +
        " · family=" + (run.family || fam) +
        " · mode=" + (run.mining_mode || "example") +
        " · scope=" + run.scope_id;
      setMsg(msg, "Streaming server-side gauntlet ceremony…", "");
      streamRun(run);
    });
  }

  function streamRun(run) {
    var collected = [];
    if (typeof EventSource !== "undefined") {
      var es = new EventSource(run.stream_url);
      state.activeSource = es;
      es.addEventListener("stage", function (ev) {
        var s = JSON.parse(ev.data);
        collected.push(s);
        renderStages(collected);
      });
      es.addEventListener("verdict", function (ev) {
        var v = JSON.parse(ev.data);
        renderVerdict(v);
      });
      es.addEventListener("done", function () {
        stopStream();
        setMsg($("mine-msg"), "Run completed and recorded in the trial ledger.", "ok");
        $("run-gauntlet-btn").disabled = !state.warehouseReady;
        $("run-example-btn").disabled = false;
        loadTrials();
        loadVault();
      });
      es.onerror = function () {
        stopStream();
        pollRun(run.poll_url);
      };
    } else {
      pollRun(run.poll_url);
    }
  }

  function pollRun(pollUrl) {
    state.pollTimer = setInterval(function () {
      api(pollUrl).then(function (r) {
        if (!r.ok || !r.data || !r.data.run) return;
        var run = r.data.run;
        renderStages(run.stages || []);
        if (run.status === "completed" || run.status === "failed") {
          stopStream();
          renderVerdict(run);
          setMsg($("mine-msg"), "Run completed and recorded in the trial ledger.", "ok");
          $("run-gauntlet-btn").disabled = !state.warehouseReady;
          $("run-example-btn").disabled = false;
          loadTrials();
          loadVault();
        }
      });
    }, 400);
  }

  $("run-gauntlet-btn").addEventListener("click", function () { startMining(false); });
  $("run-example-btn").addEventListener("click", function () { startMining(true); });

  // ---- trials ----
  function loadTrials() {
    api("/api/trials").then(function (r) {
      if (!r.ok || !r.data) return;
      var ks = r.data.kill_rate_summary || {};
      var badge = $("kill-rate-badge");
      if (ks.opt_in) {
        badge.classList.remove("hidden");
        badge.textContent = "my record: " + ks.total_trials + " trials, " + ks.killed_count + " killed";
      } else {
        badge.classList.add("hidden");
      }

      var list = $("trials-list");
      clearEl(list);
      var trials = r.data.trials || [];
      if (!trials.length) {
        list.appendChild(mk("p", { className: "fine-print serif" }, "No registered trials yet. Configure Slot A and Slot B or run the guided example."));
        return;
      }
      trials.forEach(function (t) {
        var row = mk("div", { className: "row" });
        var left = mk("div", {});
        left.appendChild(mk(
          "div",
          { className: "mono" },
          t.trial_id + " · " + t.scope_id + " · " + (t.mining_mode || "example") +
          (t.subspace_size > 1 ? (" (N=" + t.subspace_size + ")") : "") +
          (t.cached ? " (cached)" : "")
        ));
        left.appendChild(mk("div", { className: "fine-print serif", style: "margin:4px 0 0" }, t.verdict_copy || ""));
        var right = mk("div", { style: "display:flex;gap:8px;align-items:center;flex-wrap:wrap" });
        var cls = t.verdict === "killed" ? "err" : (t.survivor_class === "validated-green" ? "ok" : "warn");
        right.appendChild(mk("span", { className: "pill " + cls }, t.verdict === "killed" ? ("killed@" + (t.killed_at || "screen")) : t.survivor_class));
        if (t.private_spec_eligible) {
          var pBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Private spec");
          pBtn.addEventListener("click", function () {
            renderPrivateSpecInto(
              $("trial-private-spec-panel"),
              "/api/private/trials/" + encodeURIComponent(t.trial_id) + "/spec"
            );
          });
          right.appendChild(pBtn);
        }
        if (t.card_id) {
          var btn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Card face");
          btn.addEventListener("click", function () {
            document.querySelector('[data-app-tab="vault"]').click();
            inspectCard(t.card_id);
          });
          right.appendChild(btn);
        }
        row.appendChild(left);
        row.appendChild(right);
        list.appendChild(row);
      });
    });
  }

  $("refresh-trials-btn").addEventListener("click", loadTrials);

  // ---- vault & connections ----
  function renderCardRow(c, stateKind) {
    var row = mk("div", { className: "row clickable" });
    var title = c.card_id.slice(0, 14) + "… · " + c.scope_id + (c.ghost_no ? (" · ghost #" + c.ghost_no) : "");
    var left = mk("div", {});
    left.appendChild(mk("div", { className: "mono" }, title));
    left.appendChild(mk("div", { className: "fine-print mono", style: "margin:2px 0 0" }, c.trial_id + " · " + c.survivor_class));
    var right = mk("div", { style: "display:flex;gap:6px;align-items:center;flex-wrap:wrap" });
    right.appendChild(mk("span", { className: "tag " + tierTagClass(c.tier) }, c.tier));
    if (c.kind === "ghost") right.appendChild(mk("span", { className: "tag" }, "ghost #" + (c.ghost_no || "")));
    row.appendChild(left);
    row.appendChild(right);
    row.addEventListener("click", function () { inspectCard(c.card_id, stateKind); });
    return row;
  }

  function loadVault() {
    api("/api/vault").then(function (r) {
      if (!r.ok || !r.data) return;
      var cap = r.data.capacity || {};
      var byTier = cap.by_tier || {};
      $("vault-capacity-line").textContent =
        "Vault used: " + (cap.used || 0) +
        " · capacities (bronze " + (byTier.bronze || 2) +
        " / silver " + (byTier.silver || 4) +
        " / gold " + (byTier.gold || 6) +
        " / platinum " + (byTier.platinum || 8) + ")";
      if (r.data.hit_loop_hint) $("vault-hit-hint").textContent = r.data.hit_loop_hint;

      var vList = $("vault-list");
      clearEl(vList);
      var vaultCards = (r.data.vault || []).concat(r.data.unassigned || []);
      if (!vaultCards.length) vList.appendChild(mk("p", { className: "fine-print serif" }, "No cards in your vault yet."));
      vaultCards.forEach(function (c) { vList.appendChild(renderCardRow(c, c.vault_state)); });

      var aList = $("archive-list");
      clearEl(aList);
      var arch = r.data.archive || [];
      if (!arch.length) aList.appendChild(mk("p", { className: "fine-print serif" }, "Archive is empty. Discarded cards stay here and in the aggregate pool — never destroyed."));
      arch.forEach(function (c) { aList.appendChild(renderCardRow(c, "archive")); });

      var cList = $("connections-list");
      clearEl(cList);
      var conns = r.data.connections || [];
      if (!conns.length) cList.appendChild(mk("p", { className: "fine-print serif" }, "No saved connections yet. Save any surviving card to monitor it."));
      conns.forEach(function (cn) {
        var row = mk("div", { className: "row" });
        var left = mk("div", {});
        left.appendChild(mk("div", {}, cn.label));
        left.appendChild(mk("div", { className: "fine-print mono", style: "margin:2px 0 0" }, cn.connection_id + " · " + cn.card_id.slice(0, 14) + "…"));
        var right = mk("div", { style: "display:flex;gap:6px;align-items:center" });
        var viewBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Face");
        viewBtn.addEventListener("click", function () { inspectCard(cn.card_id); });
        var rmBtn = mk("button", { type: "button", className: "btn btn-danger btn-sm" }, "Remove");
        rmBtn.addEventListener("click", function () {
          api("/api/connections/" + encodeURIComponent(cn.connection_id), { method: "DELETE" }).then(loadVault);
        });
        right.appendChild(viewBtn);
        right.appendChild(rmBtn);
        row.appendChild(left);
        row.appendChild(right);
        cList.appendChild(row);
      });
    });
  }

  $("refresh-vault-btn").addEventListener("click", loadVault);

  function saveConnection(cardId) {
    api("/api/connections", {
      method: "POST",
      body: { card_id: cardId, notify_enabled: true }
    }).then(function (r) {
      if (r.ok) {
        setMsg($("mine-msg"), "Connection saved to your vault.", "ok");
        loadVault();
      } else {
        setMsg($("mine-msg"), (r.data && r.data.error) || "Could not save connection.", "err");
      }
    });
  }

  function inspectCard(cardId, stateKind, withImage) {
    state.selectedCardId = cardId;
    var url = "/api/cards/" + encodeURIComponent(cardId) + "/face" + (withImage ? "?with_image=true" : "");
    api(url).then(function (r) {
      var panel = $("card-face-panel");
      clearEl(panel);
      if (!r.ok || !r.data || !r.data.face) {
        var box = mk("div", { className: "panel-box" });
        box.appendChild(mk("p", { className: "form-msg err" }, (r.data && r.data.error) || "Could not load card face."));
        panel.appendChild(box);
        return;
      }
      var f = r.data.face;
      var card = mk("div", { className: "tcard t-" + (f.tier || "bronze") });
      var band = mk("div", { className: "band" });
      band.appendChild(mk("span", {}, (f.tier || "bronze") + " card" + (f.ghost_no ? (" · ghost #" + f.ghost_no) : "")));
      band.appendChild(mk("span", { className: "mono", style: "font-size:0.75rem" }, f.survivor_class || ""));
      card.appendChild(band);

      var body = mk("div", { className: "body" });
      body.appendChild(mk("div", { className: "mono fine-print", style: "margin:0 0 10px" }, f.card_id + " · " + f.family + " · " + f.scope_id));
      body.appendChild(mk("p", { className: "explainer" }, f.explainer || ""));

      // Magnitude gauge (band only by default, numeric |z| one click away)
      if (f.gauge) {
        body.appendChild(mk("div", { className: "field-label" }, "Magnitude: " + (f.gauge.label || "")));
        var g = mk("div", { className: "gauge" });
        ["compressed", "normal", "stretched"].forEach(function (b) {
          var cls = "gband" + (f.gauge.band === b ? (" on-" + b) : "");
          g.appendChild(mk("span", { className: cls }, b));
        });
        body.appendChild(g);
        if (f.gauge.magnitude_z !== null && f.gauge.magnitude_z !== undefined) {
          var zDet = mk("details", { style: "margin-top:8px" });
          zDet.appendChild(mk("summary", { className: "mono", style: "cursor:pointer;font-size:0.78rem;color:var(--accent)" }, "Show numeric |z| (magnitude only)"));
          zDet.appendChild(mk("p", { className: "mono fine-print" }, "|z| = " + f.gauge.magnitude_z + " — " + (f.gauge.note || "")));
          body.appendChild(zDet);
        }
      }

      // Health ratio phrase
      if (f.health) {
        body.appendChild(mk("p", { style: "margin:14px 0 6px;color:var(--ink)" }, "Health: " + (f.health.phrase || "")));
      }

      // Gauntlet survival facts
      if (f.gauntlet_stats) {
        var gs = f.gauntlet_stats;
        var stWrap = mk("div", { className: "stages" });
        (gs.stages_passed || []).forEach(function (s) {
          stWrap.appendChild(mk("span", { className: "stage pass" }, String(s).toUpperCase() + " pass"));
        });
        body.appendChild(stWrap);
      }

      // Actions
      var act = mk("div", { style: "display:flex;gap:8px;flex-wrap:wrap;margin:14px 0" });
      var saveBtn = mk("button", { type: "button", className: "btn btn-brass btn-sm" }, "Save connection");
      saveBtn.addEventListener("click", function () { saveConnection(f.card_id); });
      act.appendChild(saveBtn);

      if (stateKind === "archive") {
        var promBtn = mk("button", { type: "button", className: "btn btn-primary btn-sm" }, "Promote to vault");
        promBtn.addEventListener("click", function () {
          api("/api/vault/" + encodeURIComponent(f.card_id) + "/promote", { method: "POST" }).then(function () {
            loadVault();
            inspectCard(f.card_id, "vault");
          });
        });
        act.appendChild(promBtn);
      } else {
        var keepBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Keep");
        keepBtn.addEventListener("click", function () {
          api("/api/vault/" + encodeURIComponent(f.card_id) + "/keep", { method: "POST" }).then(loadVault);
        });
        var discBtn = mk("button", { type: "button", className: "btn btn-danger btn-sm" }, "Discard to archive");
        discBtn.addEventListener("click", function () {
          api("/api/vault/" + encodeURIComponent(f.card_id) + "/discard", { method: "POST" }).then(function () {
            loadVault();
            inspectCard(f.card_id, "archive");
          });
        });
        act.appendChild(keepBtn);
        act.appendChild(discBtn);
      }

      if (!withImage) {
        var imgBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Render PNG artifact");
        imgBtn.addEventListener("click", function () { inspectCard(f.card_id, stateKind, true); });
        act.appendChild(imgBtn);
      }
      body.appendChild(act);

      if (f.image && f.image.png_1x_b64) {
        body.appendChild(mk("img", {
          className: "card-png",
          alt: "Server-rendered anonymized SIFT card face",
          src: "data:image/png;base64," + f.image.png_1x_b64
        }));
      }

      // Glossary tooltips (jargon stays behind expandable disclosures)
      if (f.glossary && Object.keys(f.glossary).length) {
        var gBox = mk("div", { className: "glossary-list" });
        gBox.appendChild(mk("div", { className: "field-label", style: "margin-top:8px" }, "Glossary"));
        Object.keys(f.glossary).forEach(function (term) {
          var d = mk("details");
          d.appendChild(mk("summary", {}, term));
          d.appendChild(mk("p", {}, f.glossary[term]));
          gBox.appendChild(d);
        });
        body.appendChild(gBox);
      }

      if (state.onboarding && state.onboarding.cards_are_candidates) {
        body.appendChild(mk("p", { className: "fine-print serif" }, state.onboarding.cards_are_candidates));
      }
      body.appendChild(mk("p", { className: "fine-print" }, f.disclaimer || ""));
      card.appendChild(body);
      panel.appendChild(card);

      // Strictly separate owner-only Private Spec trigger (never inside the shareable face)
      var privWrap = mk("div", { style: "margin-top:14px" });
      var revealBtn = mk("button", { type: "button", className: "btn btn-ghost btn-sm" }, "Reveal private owner spec (symbols & parameters)");
      var privSlot = mk("div", { id: "card-private-spec-slot" });
      revealBtn.addEventListener("click", function () {
        renderPrivateSpecInto(
          privSlot,
          "/api/private/cards/" + encodeURIComponent(f.card_id) + "/spec"
        );
      });
      privWrap.appendChild(revealBtn);
      privWrap.appendChild(privSlot);
      panel.appendChild(privWrap);
    });
  }

  // ---- boot ----
  loadOnboarding().then(checkSession);
})();
