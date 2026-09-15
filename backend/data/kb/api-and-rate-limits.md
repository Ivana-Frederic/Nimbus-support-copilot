---
source: API & Rate Limits Guide
recency_score: 0.95
popularity_score: 0.85
---

# Authentication

All API requests require a Bearer token in the `Authorization` header.
Generate tokens from Settings > API Tokens. Tokens can be scoped to
read-only or read-write, and can be revoked at any time. Revoking takes
effect within a few seconds.

# Rate limits

Starter plans are limited to 1,000 requests/day (roughly 40/hour, enforced
with a sliding window). Pro plans allow 50,000 requests/day. When you exceed
your limit, the API returns HTTP 429 with a `Retry-After` header telling you
how many seconds to wait. Burst traffic within the limit is fine; sustained
overuse for multiple days may trigger a temporary throttle even on Pro.

# Common errors

- `401 Unauthorized`: token missing, expired, or revoked.
- `403 Forbidden`: token doesn't have the required scope for this endpoint.
- `429 Too Many Requests`: rate limit exceeded, see above.
- `500 Internal Server Error`: an unexpected server-side error. These are
  logged automatically; if you see repeated 500s, include the
  `X-Request-Id` response header when contacting support.

# Webhooks

Webhooks fire for `object.created`, `object.updated`, and `object.deleted`
events. If a webhook delivery fails, we retry with exponential backoff for
up to 24 hours. You can inspect delivery attempts and replay failed
deliveries from Settings > Webhooks > Delivery Log. Make sure your endpoint
returns a 2xx status within 10 seconds or the delivery is considered failed.

# SDKs

Official SDKs are available for Python, Node.js, and Go. All three wrap the
same REST API and handle retries/backoff automatically for 429 and 5xx
responses.
