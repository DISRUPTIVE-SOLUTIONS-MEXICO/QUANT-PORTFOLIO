# Security Threat Model

## Assets

- Supabase service-role credential and user JWTs.
- Immutable research and portfolio artifacts.
- User suitability profiles, portfolio versions and paper orders.
- Strategy Constitution, model registry and promotion evidence.
- Private/MNPI information, which must remain outside shared workflows.

## Trust Boundaries

| Boundary | Control |
|---|---|
| Public source -> worker | validation, provenance, hashes, freshness, fallbacks |
| Browser -> API | Supabase JWT verification, Zod input validation, rate limits |
| API -> Supabase | RLS for user reads, service role only on trusted server paths |
| Worker -> publication | Pydantic contracts, SHA-256, staging, quality tests |
| User -> paper order | suitability, promotion, OOS, liquidity, cost, human approval |
| Electron renderer -> OS | sandbox, context isolation, no Node, URL allowlist |

## Primary Threats

1. Cross-user portfolio disclosure.
2. Service-role key inclusion in a browser bundle.
3. Artifact substitution or partial publication.
4. Promotion of research-only evidence.
5. Stale prices entering a paper order.
6. Excess ADV participation or concentration.
7. Multiple-testing evidence hidden from users.
8. MNPI leaking into shared signals or assistant context.
9. Malicious public-data payloads or schema drift.
10. Electron navigation to an untrusted origin.

## Mandatory Controls

- Backend-only writes for portfolio versions, order intents and decisions.
- Owner-read RLS for all user resources.
- Atomic pointers by publication kind and tenant.
- Canonical JSON hashes checked before pre-trade evaluation.
- No paper submission without a named human approver.
- No real broker connection in V1.
- Secret scanning and dependency auditing in CI.
- Audit events are append-only and contain no raw secret values.

## Accepted Residual Risks

- Public data are research-grade, not vendor-grade PIT.
- Free sources may fail, lag or change schemas.
- The current stable Next.js release pins a PostCSS version with a moderate
  build-time advisory. The build does not process user-supplied CSS; CI blocks
  high/critical advisories and this exception must be reviewed at each upgrade.
