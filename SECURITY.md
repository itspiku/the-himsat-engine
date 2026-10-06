# Security policy

HimSat is part of an early-warning chain. A compromised deployment could suppress real warnings or
send false ones, so please report vulnerabilities privately.

## Reporting

Use GitHub's private vulnerability reporting (**Security → Report a vulnerability**) on this
repository. Do not open public issues for security problems. Include the affected version, the
impact and steps to reproduce. Maintainers aim to acknowledge reports promptly.

## Deployment hardening checklist

- Terminate TLS in front of the API. Expose only port 443.
- Admin API keys: generate with `himsat admin new-key`, store only the SHA-256 hash in
  `HIMSAT_ADMIN_API_KEYS`, rotate on staff changes. Rate-limit `/api/admin` at the proxy.
- Set `HIMSAT_CORS_ORIGINS` to your own domain (the default `*` is for development only).
- Set `HIMSAT_WEBHOOK_SECRET`. Receivers must verify the `X-HimSat-Signature` HMAC-SHA256 header.
- Keep `HIMSAT_AUTO_DISPATCH_LEVELS=high` (or empty) so lower-confidence alerts get human review.
- Run the LLM server on a private network. HimSat sends it only the alert fact sheet, never
  subscriber data.
- Back up PostgreSQL daily. The `alerts`, `deliveries` and `risk_assessments` tables are the audit
  trail.
- Containers run as an unprivileged user (`uid 10001`) with a single writable volume (`/data`).
