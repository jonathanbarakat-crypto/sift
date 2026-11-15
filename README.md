# SIFT

SIFT mines cross-stock relationship patterns server-side and tests them against a
pre-registered, multiplicity-aware statistical gauntlet. **SIFT does not trade.**

Public content shows survival facts only — no returns, no win rates, no directional calls.
No raw market data is ever served to clients; browsers receive derived outputs only.

## Repository layout

```
web/                  static landing page (plain HTML/CSS/JS, no framework)
  index.html          landing page
  styles.css          styles
  app.js              waitlist form handling (progressive enhancement)
  api/waitlist.py     optional stdlib-only server: static host + POST /api/waitlist
  api/test_waitlist.py  stdlib unittest suite for the endpoint (run below)
data/                 waitlist storage (waitlist.csv, created on first signup; gitignored)
```

## Serving the landing page

The page is plain HTML/CSS/JS and deploys to **any static host** as-is
(GitHub Pages, Netlify, Cloudflare Pages, S3, nginx, ...). From this repo:

```bash
# Option A — any static server pointed at web/
python3 -m http.server 8080 --directory web
# then open http://127.0.0.1:8080/

# Option B — the bundled stub, which also provides the waitlist endpoint
python3 web/api/waitlist.py            # http://127.0.0.1:8080/
PORT=9000 python3 web/api/waitlist.py  # custom port
```

### Waitlist endpoint

The form on the page POSTs to `/api/waitlist` with `{"email": "..."}`
(JSON, or form-encoded `email=...` as fallback). The bundled stub
(`web/api/waitlist.py`, Python standard library only, no dependencies)
accepts it and appends to `data/waitlist.csv` (created on first signup,
with header `email,ts_utc,source`; duplicates are ignored).

On a pure static host without the stub, map any tiny endpoint to
`POST /api/waitlist` that accepts the same payload and persists the email —
or run the stub behind any reverse proxy. Sign-ups are email-only.

The stub is a dev tool: single process, plain HTTP, no auth — put it behind a
reverse proxy (TLS + rate limiting) in any real deployment. It writes one
append-only CSV (`data/waitlist.csv`, gitignored), guards the body size,
blocks path traversal, never serves `.py`/`.csv`/`.json`/`.md` under `web/`,
neutralizes spreadsheet formula prefixes (`=`, `+`, `-`, `@`) in stored
emails, and serializes writes so concurrent signups can't duplicate rows.

Tests (stdlib only):

```bash
python3 -m unittest discover -s web/api -p "test_*.py" -v
```

## Beta

The first beta cohort is 50–100 quantitative traders, invited in waves, free
during the beta.
