# Contributing to HimSat Engine

HimSat issues warnings that people may act on. Changes are welcome. Anything that can change what
gets detected or what an alert says gets extra scrutiny.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                           # add ,ml for Prithvi/SAM; ,insar for HyP3
pytest -q -m "not network and not gpu"            # fast, offline test suite
ruff check himsat tests
cd web && npm install && npm run build            # type-check + build the map
```

## Rules for changes that affect warnings

1. **Detection or risk logic** (`himsat/detect`, `himsat/risk`, `config/risk.yaml`): bump
   `model_version` in `risk.yaml`. Rerun the reference hindcast and include the before/after alert
   counts and lead times in the PR:
   ```bash
   himsat hindcast rasuwa-lhende --start 2026-03-01 --end 2026-09-10 \
     --event-time 2026-08-26T02:52:00+00:00 --event-lon 85.5282 --event-lat 28.2881 --name pr-check
   himsat reassess rasuwa-lhende --hindcast pr-check   # after risk-only changes (no imagery reprocessing)
   ```
   False alarms count as much as missed events: report both.
2. **Alert wording** (`himsat/alerts/phrases.py`, `templates.py`): Nepali changes need review by a
   native speaker from the disaster-risk-reduction community. Templates must keep passing
   `compose.validate` (tests enforce this).
3. **LLM prompts or validator**: never loosen a validator check without a test showing why it was
   wrong. The validator is the safety barrier between a language model and the public.
4. **New data sources**: document licence and attribution in `docs/DATA_LICENSES.md`.

## Style

- Python 3.11, type hints, `ruff` clean, docstrings that explain *why* a threshold exists.
- Keep physics and statistics explicit and configurable, not hidden in code.
- Tests must run offline. Mark tests that need the network with `@pytest.mark.network`.

## Commit messages

Imperative, specific subject lines ("Require cross-orbit confirmation for hotspots"), with a body
explaining the evidence for detection changes.
