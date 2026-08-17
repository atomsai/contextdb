## Summary

<!-- What changed, and why. Evals win over prose. -->

## Open-core checklist

- [ ] I have read OPEN_CORE.md. This PR does not add cloud, billing, SSO, or entitlement code to the Apache tree.
- [ ] The SDK still does not import `contextdb_cloud` or any cloud control plane.
- [ ] No new local paid / license-key / entitlement boolean.
- [ ] Substantial contribution: I agree to the CLA in CONTRIBUTING.md (legal name and email in this description). Trivial typo PRs may skip this.
- [ ] `pytest tests/evals/ -v` and `python scripts/check_open_core.py` pass, or I explain why not.
