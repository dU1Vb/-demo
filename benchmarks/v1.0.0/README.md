# Benchmark Library v1.0.0

This directory materializes the static TorchBridgeBench benchmark catalog
described in `ClaudeCodePluginDesign.md` Appendix A.

- Total cases: `175`
- L1 cases: `67`
- L2 cases: `42`
- L3 cases: `25`
- L4 cases: `41`

Generation command:

```bash
python scripts/generate_benchmark_library.py
```

The generator is build-time only. Runtime evaluation still consumes the static
JSON cases under `cases/`.

Bundled suites:

- `suites/dev_noop.json`: fast routine validation.
- `suites/smoke_noop.json`: cross-level benchmark smoke including L3/L4.
- `suites/all_noop.json`: full generated matrix.
