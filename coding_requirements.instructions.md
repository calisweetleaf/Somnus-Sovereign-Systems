---
applyTo: "*Agent, Chat, Edit*" 
---

# Ultra-Strict Production Coding Requirements

These requirements are binding for all code, languages, frameworks, environments, and contributors to this project. Non-compliant contributions must be rejected.

## Objectives

- Generate production-ready, secure, high-performance, maintainable code with zero stubs, zero dead code, and verifiable quality.
- Enforce measurable, automated quality gates at every change.

## Definition of Done (DoD)

A change is eligible to merge only if ALL of the following are true:

1. Functional completeness: No stubs, placeholders, pseudo-code, or commented-out code paths. No TODO/FIXME.
2. Tests:
   - Unit coverage ≥ 90% lines and ≥ 90% branches for changed files.
   - Mutation testing score ≥ 75% on changed files or rationale with risk acceptance by code owners.
   - Integration tests cover critical paths and error scenarios.
   - E2E tests updated if user-facing behavior changes.
   - Flaky test budget = 0; tests must be deterministic.
3. Static quality gates (CI-enforced):
   - Lint: 0 errors; warnings must be justified or fixed.
   - Type-check: 0 errors.
   - Complexity thresholds: cyclomatic ≤ 10 per function unless justified; functions > 50 LOC must be refactored or justified.
   - Duplication: no copy-paste beyond 3 lines without abstraction.
4. Security:
   - Secrets never hardcoded; use secure config/secret stores.
   - SAST: 0 high/critical findings; medium requires documented mitigation and code owner approval.
   - DAST (where applicable): 0 high/critical.
   - SCA: 0 vulnerable dependencies at high/critical; pinned versions or lockfiles required.
   - Threat model updated for new threat surfaces (STRIDE or equivalent) with mitigations.
   - Input validation, output encoding, and least privilege enforced. No direct string concatenation for queries; use parameterization.
5. Performance and reliability:
   - Performance SLOs defined for changed components with benchmarks or load tests for hot paths. Regressions > 2% require approval and tracking.
   - Memory safety and bounded resource usage; no unbounded growth or leaks.
   - Concurrency safety: no data races; proper synchronization or immutability.
   - Timeouts, retries with backoff, and circuit breakers for network/IO.
6. Observability and operations:
   - Structured logs with correlation IDs; no sensitive data in logs.
   - Metrics (latency, errors, resource) and health checks for services.
   - Tracing added for critical spans.
   - Feature flags for risky changes; documented rollout/rollback.
7. Accessibility and UX (for UI):
   - WCAG 2.1 AA: keyboard navigability, contrast, ARIA semantics, focus management.
   - Screen reader support validated.
8. Privacy and compliance:
   - Data classification applied; PII/PHI handled per policy.
   - Data minimization, purpose limitation, retention documented.
   - Redaction for logs/exports; encryption in transit and at rest where applicable.
9. Documentation:
   - Public APIs/classes/functions have docstrings with parameters, returns, errors, examples.
   - Architectural decision record (ADR) for non-trivial changes.
   - Update READMEs, runbooks, and migration notes.
10. Code review and governance:

- At least 2 approving reviews from code owners for critical areas; 1 for others.
- No self-approval; no force merges.
- Commit messages: imperative, reference issue/ADR, describe rationale and impact.
- Trunk-based development with short-lived branches; CI green required.

11. Backwards compatibility:

- Avoid breaking changes; if unavoidable, provide migrations, deprecation schedule, and clear release notes.

12. Internationalization (if applicable):

- No hardcoded user-facing strings; use i18n framework and pluralization rules.

## Mandatory Practices

- Full implementation only; partial solutions are rejected.
- Defensive programming: validate inputs, handle all error paths, and fail fast with clear messages.
- Idempotency for mutation endpoints and jobs where applicable.
- Resource cleanup and cancellation support.
- Use established patterns: dependency injection, single responsibility, clear boundaries.
- Interfaces and contracts are explicit; avoid hidden side-effects.

## Tooling and Automation (CI Quality Gates)

- Linting and formatting: enforced via CI. Config checked into repo.
- Type checking: enforced with strict mode where available.
- SAST: run on every PR (e.g., CodeQL/Bandit/ESLint security).
- DAST: run on protected branches or preview envs for services with HTTP surfaces.
- SCA and license compliance: fail on incompatible licenses or high/critical CVEs; use allowlists/overrides with owner sign-off.
- IaC scanning for Terraform/K8s/Cloud configs; 0 criticals.
- Container checks: minimal base images, non-root user, pinned tags/digests, vulnerability scan gates.
- Mutation testing for critical libraries/services on changed files.
- Test flake detection and quarantine require issue creation and fix before merge.

## Performance and Load Testing

- Provide micro-benchmarks for algorithms and hot paths.
- For services: load tests with representative traffic; define throughput/latency/error budgets. Document baseline and delta.
- Use profiling to justify algorithmic choices when complexity > O(n log n) or large constants.

## Error Handling and Resilience

- No silent failures. Log at appropriate levels with actionable context.
- Wrap external calls; implement retries, timeouts, and jittered backoff.
- Graceful degradation and circuit breakers under partial outages.
- Dead-letter queues or retries for async processing with observability.

## Security Best Practices (Non-Exhaustive)

- Use parameterized queries; no dynamic SQL string concatenation.
- Validate and sanitize all external inputs; whitelist preferred over blacklist.
- Escape/encode outputs to prevent XSS/HTML injection.
- CSRF protection for state-changing HTTP endpoints.
- Strong password policies; modern KDFs (e.g., Argon2/bcrypt/scrypt) with salt and appropriate cost.
- JWT/session hardening: short TTLs, rotation, audience/issuer checks.
- Enforce least privilege IAM and scoped tokens.
- Regular key rotation; secrets from env/secret manager, not from code or VCS.
- Logging excludes secrets, tokens, credentials, and sensitive identifiers.

## Maintainability and Readability

- Small, cohesive modules and functions with clear names.
- Avoid deep inheritance; prefer composition.
- Public APIs stable and documented; internal details hidden.
- Comments explain why, not what; code should be self-explanatory.
- No premature optimization; measure and document when optimizing.

## Acceptance Checklist (must be met before merge)

- [ ] No partial implementations; no TODO/FIXME.
- [ ] All inputs validated; all error paths handled; resources cleaned up.
- [ ] Style/lint/type checks pass with 0 errors.
- [ ] Line and branch coverage ≥ 90% for changed files; mutation ≥ 75% or approved exception.
- [ ] Integration/E2E tests updated; no flakiness.
- [ ] SAST/SCA/IaC/container scans: 0 high/critical; mediums documented or fixed.
- [ ] Performance baselines present; no >2% regression without approval.
- [ ] Observability (logs/metrics/traces/health) implemented.
- [ ] ADRs/docs/runbooks updated.
- [ ] Accessibility (WCAG 2.1 AA) verified where applicable.
- [ ] Privacy/compliance reviewed; no sensitive data leakage.
- [ ] Reviews completed per code owners; CI green.
- [ ] Rollout/rollback plan documented; feature flags where needed.

## Immediate Escalation

If any requirement cannot be met:

1. Halt coding.
2. Document the blocker with evidence.
3. Notify code owners and request guidance or a risk acceptance decision.

Non-compliance will result in immediate rejection of code changes.
