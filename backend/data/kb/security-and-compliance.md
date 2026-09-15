---
source: Security & Compliance Overview
recency_score: 0.7
popularity_score: 0.5
---

# Compliance certifications

Nimbus Cloud is SOC 2 Type II certified (report available under NDA via
Settings > Compliance) and is in the process of ISO 27001 certification.
We are GDPR-compliant for customers processing EU personal data.

# Encryption

All data is encrypted in transit with TLS 1.2+ and at rest with AES-256.
Encryption keys are managed via a hardware security module (HSM) and
rotated automatically every 90 days.

# Authentication and access control

Two-factor authentication (2FA) can be enabled from Settings > Security and
is required for all Enterprise plan admins. Enterprise plans also support
SAML-based single sign-on (SSO) with most identity providers (Okta, Azure
AD, Google Workspace). IP allow-listing for admin console access is
available on Enterprise plans.

# Reporting a security issue

If you believe your account has been compromised, reset your password
immediately from Settings > Security > Reset Password and revoke all active
API tokens. For suspected vulnerabilities in our platform, email
security@nimbuscloud.example. We do not currently run a public bug bounty
program, but we do acknowledge and credit responsible disclosures.

# Data deletion requests

GDPR/CCPA data deletion requests can be submitted from Settings > Privacy >
Delete My Data, or by emailing privacy@nimbuscloud.example. Deletion
requests are processed within 30 days and are irreversible.
