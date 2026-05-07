# Repository Guidelines

## Project Shape

Agent Wrangler is a macOS/tmux control layer for managing multiple AI coding agents. The public CLI entry point is `scripts/agent-wrangler`, which routes to the Python implementation in `scripts/agent_wrangler.py` and shared helpers under `scripts/aw/`.

## Commands

```bash
./scripts/agent-wrangler status
python3 -m pytest tests/ -v
```

## Change Rules

- Preserve zero-install behavior: stdlib Python and Bash only for runtime code.
- Keep macOS/Ghostty-specific behavior explicit in docs and tests.
- Do not commit local runtime state from `.state/`, `reports/`, or `config/projects.json`.
- Update `README.md` when user-facing commands, shortcuts, or setup requirements change.
