# Deploy guide — Quant Portfolio-Kaizen

Zero-cost, authenticated, with CI quality gates.

## TL;DR

- **Cheapest path to production**: push to GitHub, click "Deploy" on Streamlit
  Community Cloud, paste secrets, done.
- **Auth + RBAC**: provided by `auth.py` (bcrypt + JWT cookies + lockout +
  audit log). Generate hashes with `python scripts/hash_password.py`.
- **CI**: `.github/workflows/ci.yml` runs tests, ruff, bandit, pip-audit, and
  Docker build on every push.
- **Hardening**: CSP / no-sniff / referrer policy / noindex from
  `security.py:inject_security_headers()` plus Streamlit XSRF and rate
  limiter.
- **Vercel**: still not recommended — see end of document.

---

## 1. Streamlit Community Cloud (recommended, $0)

### Prerequisites

- GitHub account.
- Repo with this code (private or public).
- A workstation with Python ≥ 3.11.

### Step 1 — generate cookie key and password hashes

```powershell
# Cookie signing key (paste into [auth] cookie_key).
python -c "import secrets; print(secrets.token_urlsafe(48))"

# Per-user password hash (paste into [auth.users.<name>] password_hash).
pip install -r requirements.txt
python scripts/hash_password.py
```

> **Dependency pinning.** `requirements.txt` keeps compatible ranges (what
> Streamlit Community Cloud installs); `requirements.lock.txt` is the
> uv-generated universal lock that CI and GitHub Actions research jobs
> install for reproducibility. Regenerate it after changing requirements:
> `uv pip compile requirements.txt --universal -o requirements.lock.txt`.

### Step 2 — push the repo

```powershell
git init
git remote add origin https://github.com/<you>/<repo>.git
git add .
git commit -m "Initial deploy"
git push -u origin main
```

The included `.gitignore` already excludes `secrets.toml`, `audit.jsonl`,
caches, and `.env` files.

### Step 3 — deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io>.
2. Sign in with GitHub and click **"New app"**.
3. Pick the repo, branch `main`, and the entry point `stockpicker_app.py`.
4. Click **"Advanced settings"**, set Python version `3.11` or `3.12`.
5. Click **"Deploy"**. The first build pulls dependencies (~3 minutes).

### Step 4 — paste secrets

After the first deploy:

1. Open the app, click `⋮` (top-right) ➜ **Settings** ➜ **Secrets**.
2. Copy the contents of `.streamlit/secrets.toml.example` into the editor.
3. Replace every `REPLACE_…` placeholder with real values (cookie key, user
   entries with bcrypt hashes from step 1, optional BANXICO / SEC user-agent
   strings).
4. Save. Streamlit redeploys automatically; the login form will appear.

### Step 5 — first login

- Username = the key under `[auth.users.<key>]` (lower-cased).
- Password = the plaintext you typed into `scripts/hash_password.py`.
- After 5 failed attempts the username is locked out for 15 minutes.

### Built-in limits of the free tier

- 1 GB RAM, shared CPU. Cold-start ~5 s after 12h idle.
- Public URLs only (cannot remove the `*.streamlit.app` domain on free tier).
- For private apps, invite GitHub usernames via the app settings; Streamlit
  enforces the SSO check **before** our auth layer runs.

---

## 2. Hugging Face Spaces ($0, more RAM)

If 1 GB is tight, Spaces gives 16 GB on the free CPU tier.

1. Create a new Space at <https://huggingface.co/new-space>, pick "Streamlit"
   as SDK.
2. Push this repo:

   ```powershell
   git remote add hf https://huggingface.co/spaces/<you>/<space-name>
   git push hf main
   ```

3. Go to the Space ➜ Settings ➜ **Repository secrets** and add the same
   key/value pairs that you would put under `[auth]` in `secrets.toml`. Use
   nested-key syntax (Spaces converts dot-separated keys to TOML tables for
   `st.secrets`):

   ```
   auth.cookie_name              qpk_auth
   auth.cookie_key               <generated>
   auth.cookie_expiry_days       1
   auth.users.chris.name         Chris
   auth.users.chris.email        chris@example.com
   auth.users.chris.password_hash $2b$12$...
   auth.users.chris.role         admin
   ```

---

## 3. Self-hosted Docker (Render / Fly / Railway / Cloud Run)

Use the included `Dockerfile`. Build and run:

```powershell
docker build -t quant-portfolio-kaizen .
docker run --rm -p 8501:8501 `
  -e STREAMLIT_SERVER_HEADLESS=true `
  -v ${PWD}/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro `
  quant-portfolio-kaizen
```

Mount `secrets.toml` as a read-only volume; the file is gitignored.

### Render (free, sleeps after 15 min idle)

1. Sign in to <https://render.com>.
2. **New ➜ Web Service ➜ Docker**, pick the repo.
3. Add the same secrets via the **Environment** tab (one variable per
   `auth.users.*` entry; Render does not parse TOML, so write a small
   `entrypoint.sh` that materialises `secrets.toml` at boot, or store the
   whole TOML blob under a single secret and `echo "$SECRETS_TOML" >
   .streamlit/secrets.toml`).
4. Health check path: `/_stcore/health`.

---

## 4. CI quality gates (already wired)

`.github/workflows/ci.yml` runs on every push / PR:

| Job | Tools | Behavior |
|---|---|---|
| `test` | `unittest` | Matrix on Python 3.11 + 3.12. Failures block merges. |
| `lint` | `ruff check`, `ruff format --check` | Currently soft-fails (`--exit-zero`); switch to hard-fail once the codebase is fully linted. |
| `security` | `bandit`, `pip-audit` | Bandit fails on medium+ severity findings (B101/B404 skipped). `pip-audit` runs and is informational. |
| `build-container` | `docker build` | Validates that the Docker image still builds. Cached via GitHub Actions cache. |

Run the same suite locally:

```powershell
pip install -r requirements-dev.txt
python -m unittest discover tests
ruff check .
bandit -r . --severity-level medium --skip B101,B404 -x ./tests,./paper,./scripts
pip-audit -r requirements.txt
```

---

## 5. Pre-deploy checklist

- [ ] `python -m py_compile stockpicker_app.py quant_stockpicker_core.py` ➜ OK.
- [ ] `python -m unittest discover tests` ➜ 54/54 OK.
- [ ] `pip install -r requirements.txt` succeeds on a clean venv.
- [ ] Generated a real `cookie_key` (≥ 32 chars) and bcrypt hashes for every user.
- [ ] `.streamlit/secrets.toml` is **not** committed (check `git status`).
- [ ] `.gitignore` covers `secrets.toml`, `audit.jsonl`, `.env*`.
- [ ] At least one `admin`, one `analyst`, one `viewer` user in secrets.
- [ ] Visited the deployed app, logged in, ran the pipeline, signed out.
- [ ] Confirmed RBAC: a `viewer` cannot see `Advanced Research`.
- [ ] Confirmed rate limiter: 6 consecutive Run clicks block the 7th.
- [ ] Confirmed lockout: 5 wrong passwords lock the username for 15 min.

---

## 6. Why Vercel is still not the default

Vercel is optimised for serverless functions and Next.js. Streamlit needs:

- a persistent Python process,
- WebSocket sessions open for the lifetime of a user session,
- a writable working directory for the parquet cache,
- run durations of several minutes for the causal pipeline.

Vercel's standard runtime caps execution time at 30 seconds and does not keep
WebSockets open across function invocations. The only way to make it work is
the Docker runtime (paid) — and even then the edge proxy will terminate the
WebSocket once it exceeds the configured timeout.

If you must ship to Vercel:

```json
// vercel.json
{
  "version": 2,
  "builds": [{ "src": "Dockerfile", "use": "@vercel/docker" }]
}
```

```powershell
npm i -g vercel
vercel deploy --prod
```

Expect: the pipeline button will time out on long backtests. Mitigations: run
the heavy backtest from a scheduled GitHub Action that writes results to
Supabase, and use the Vercel-hosted Streamlit purely as a render layer that
reads from Supabase.
