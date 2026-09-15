---
source: Outages & Incident History
recency_score: 1.0
popularity_score: 0.6
---

# How to check current status

Live status for all Nimbus Cloud regions is published at
status.nimbuscloud.example (uptime, active incidents, and scheduled
maintenance). You can also subscribe to incident email/SMS alerts from that
page.

# What counts as an outage

We classify an incident as a partial outage when a subset of customers or
regions are affected, and a full outage when the platform is unreachable
globally. Elevated error rates or latency above our SLO for more than 5
minutes are reported as "degraded performance" incidents, even if the
service is technically reachable.

# What to do during an incident

Retry requests with exponential backoff. Do not hammer the API during an
active incident, since that slows recovery for everyone. Check the status page
for updates before contacting support; incident updates are typically
posted every 15-30 minutes until resolution.

# Post-incident reports

For any incident lasting longer than 30 minutes, we publish a post-incident
report within 5 business days covering root cause, impact, and remediation
steps, linked from the status page's incident history.

# SLA credits

Enterprise customers with an uptime SLA can request service credits for
incidents that breach the agreed uptime threshold by emailing
billing@nimbuscloud.example with the incident id from the status page.
