# Security policy — Quant Portfolio-Kaizen

## Supported versions

The app is shipped from `main`. Only the latest commit is supported. Pinned
production tags (`v*`) receive security-only backports for 90 days from release.

## Reporting a vulnerability

Email **security@<your-domain>** with:

- A clear reproduction (curl / Python snippet preferred).
- The affected commit SHA or deployment URL.
- Impact assessment (data exposure, RCE, auth bypass, etc.).

Coordinated disclosure: we acknowledge within 72h and aim to ship a fix or
mitigation within 14 days for critical issues. **Do not file a public issue or
PR for security bugs.**

## Threat model

| Asset | Threat | Mitigation |
|---|---|---|
| User credentials | Brute-force, credential stuffing | bcrypt hashes (cost 12), per-username lockout (5 attempts / 15 min), audit log |
| Session cookie | Theft via XSS / MITM | JWT signed with a 32+ char secret, `HttpOnly` + `Secure` set by platform TLS, short expiry (1 day default) |
| Backend compute | Accidental DoS by re-running expensive pipeline | Per-session rate limiter: 6 runs / 15 minutes |
| Inline HTML rendering | Reflected XSS via injected ticker / portfolio fields | `html.escape` on every untrusted string fed into pills/banners/headers; payload contract enforces `Gate_Status ∈ {approved, blocked}` |
| Secrets at rest | Disclosure via repo / logs | `.gitignore` excludes `secrets.toml`, `.env`, `audit.jsonl`; Streamlit Cloud encrypts secrets at rest |
| Data exfiltration | Read-only role downloading internals | RBAC limits `viewer` to Overview / Allocation / Price Path; Advanced Research is `admin`-only |
| Supply chain | Compromised dependency | `pip-audit` runs on every PR; `requirements.txt` is the only install path; CI builds the Docker image to detect breakage |
| Click-jacking | Embedded in attacker iframe | CSP `frame-ancestors 'none'` + `X-Frame-Options` (set by Streamlit platform); content `noindex, nofollow` meta tag |
| Sensitive logging | PII in logs | Audit log records `username`, role, action, duration — never password, never secrets |

## Defense-in-depth layers (in order)

1. **Network**: HTTPS terminated by Streamlit Cloud / the container platform.
2. **Application**:
   - Document-level meta hardening (`referrer-policy`, `robots: noindex`) injected from `security.py:inject_security_headers()`.
   - **Content-Security-Policy** must be set as an HTTP response header at the
     platform/proxy layer (Streamlit Cloud, Render, Fly, Cloud Run all expose
     this). Meta-CSP is intentionally **not** injected because Streamlit's
     bundle uses dynamic style/font/blob URLs that get false-positive blocked.
     Recommended header:
     `default-src 'self' data: blob:; img-src 'self' data: blob: https:;
     style-src 'self' 'unsafe-inline' https://fonts.googleapis.com data:;
     font-src 'self' https://fonts.gstatic.com data: blob:; script-src 'self'
     'unsafe-inline' 'unsafe-eval' blob:; connect-src 'self' wss: https:;
     frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'`
   - XSRF on (Streamlit config).
3. **Authentication**: `auth.py:require_authentication()` blocks every code path
   below it via `st.stop()` if the session is unauthenticated.
4. **Authorization (RBAC)**: section-level checks via `AuthenticatedUser.can_access()`.
5. **Rate limiting**: `security.py:RateLimiter` per-user, sliding window.
6. **Input validation**: `security.py:sanitize_ticker(_list)` and
   `sanitize_free_text` for every external string.
7. **Output safety**: `_status_pill`, `_section_header`, `_empty_state`,
   `_banner` all `html.escape` user-controlled strings before rendering.
8. **Audit**: `audit.jsonl` append-only JSON Lines. Rotate via cron / platform.

## Roles

| Role | Sections available | Use case |
|---|---|---|
| `admin` | All including Advanced Research, Data Freshness | Quant lead, ops |
| `analyst` | Overview, Allocation, Private Alpha, Price Path, Risk, Validation, Market Regime, Options, Fundamentals, Data Freshness | Portfolio analyst |
| `viewer` | Overview, Allocation, Price Path | Stakeholder / read-only |

## Operational hardening

- **Cookie key rotation**: change `auth.cookie_key` in secrets every 90 days.
  Active sessions are invalidated immediately.
- **Password rotation**: enforce via policy (the app does not enforce expiry).
- **Audit retention**: ship `audit.jsonl` to S3 / Supabase weekly; the local
  file is ephemeral on Streamlit Cloud.
- **Dependency updates**: enable Dependabot on the repo (GitHub UI ➜ Insights
  ➜ Dependency graph ➜ Dependabot alerts).
- **Container scan**: GitHub Actions builds the Docker image on every PR;
  consider adding Trivy as an extra job for image CVE scanning.

## Out of scope

- DDoS protection beyond the platform's default rate limits.
- Account self-service (registration, password reset by email). Both are
  intentionally disabled; users are provisioned by an administrator who edits
  the secrets file.
- Multi-factor authentication. Streamlit Community Cloud supports OAuth /
  Google / GitHub SSO natively — if you need MFA, deploy with native auth
  instead of the bcrypt path (see `DEPLOY.md`).
