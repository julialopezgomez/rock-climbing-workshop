# Saved reproduction outputs

These files were produced by this checkout's runtime on the included saved RGB
cases. They are inspection records, not inference inputs. `verification.json`
in the parent directory records commands, environment and scope.

- `holds/`: exact source hold records and equality checks.
- `robot/`: selected configured estimator, raw diagnostics and post-run scores.
- `trunk/`: image-only root estimates; no joint or endpoint reconstruction.
- `combined/`: fast robot/cup replay. Candidate counts are not new accuracy scores.

Diagnostic timings and temporary native/cache paths describe the recorded
machine. They are not runtime dependencies. GPU fitting may differ slightly
between runs; use the stated pose thresholds. Exact recovered hold dictionaries
are compared without numerical tolerance.
