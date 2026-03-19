# Walk-Away Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable dispatching agents across projects and walking away, with a briefing command to catch up when you return.

**Architecture:** Add an activity log that persists state transitions to disk during each rail refresh cycle. Add stall detection to distinguish "just finished" from "stuck". Add a `briefing` command that reads the log and summarizes what happened. Ensure all agent launch paths default to auto-pilot mode.

**Tech Stack:** Python 3.10+ stdlib only (json, pathlib, time, dataclasses). No new dependencies.

---

### Task 0: Commit the barn feature

The barn discovery + click-to-graze feature is sitting uncommitted. Commit it before starting new work.

**Step 1: Commit the current changes**

```bash
git add scripts/agent_wrangler.py scripts/welcome_banner.sh docs/plans/2026-03-14-barn-discovery-graze-design.md
git commit -m "feat: barn discovery + click-to-graze in rail"
```

---

### Task 1: Activity log infrastructure

Write state transitions to `.state/activity.jsonl` during each rail refresh. One JSON object per line, appended. This is the foundation for briefing and stall detection.

**Files:**
- Modify: `scripts/agent_wrangler.py` (add activity log functions near health state code, ~line 3164)

**Step 1: Add activity log constants and writer**

Add after the `NOTIFY_STATE_PATH` block (~line 3164):

```python
ACTIVITY_LOG_PATH = ROOT / ".state" / "activity.jsonl"
ACTIVITY_MAX_BYTES = 5 * 1024 * 1024  # 5 MB, then rotate


def _append_activity(entries: list[dict[str, Any]]) -> None:
    """Append activity entries to the JSONL log. Auto-rotates at 5 MB."""
    if not entries:
        return
    ACTIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        if ACTIVITY_LOG_PATH.exists() and ACTIVITY_LOG_PATH.stat().st_size > ACTIVITY_MAX_BYTES:
            rotated = ACTIVITY_LOG_PATH.with_suffix(".jsonl.old")
            ACTIVITY_LOG_PATH.rename(rotated)
    except OSError:
        pass
    try:
        with ACTIVITY_LOG_PATH.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _read_activity(since_minutes: float = 0, limit: int = 500) -> list[dict[str, Any]]:
    """Read activity entries, optionally filtered to the last N minutes."""
    if not ACTIVITY_LOG_PATH.exists():
        return []
    cutoff = 0.0
    if since_minutes > 0:
        cutoff = time.time() - (since_minutes * 60)
    entries: list[dict[str, Any]] = []
    try:
        with ACTIVITY_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff and entry.get("ts", 0) < cutoff:
                    continue
                entries.append(entry)
                if len(entries) > limit:
                    entries = entries[-limit:]
    except OSError:
        pass
    return entries
```

**Step 2: Verify it works**

Run: `python3 -c "import sys; sys.path.insert(0, 'scripts'); from agent_wrangler import _append_activity, _read_activity, ACTIVITY_LOG_PATH; _append_activity([{'ts': 1, 'test': True}]); print(ACTIVITY_LOG_PATH.exists()); print(_read_activity())"`

Expected: `True` and a list with one entry. Then delete the test file: `rm .state/activity.jsonl`

**Step 3: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: activity log infrastructure (JSONL append + reader)"
```

---

### Task 2: Log state transitions in the rail loop

Hook into the rail refresh cycle to log health transitions. Only log when a pane's state actually changes (not every 5-second tick).

**Files:**
- Modify: `scripts/agent_wrangler.py` (add tracking state + logging call in `_rail_loop`)

**Step 1: Add transition tracking state**

Add to the module-level state variables near line 88 (where `_health_history` etc. are defined):

```python
_prev_activity_state: dict[str, dict[str, Any]] = {}  # project_id -> last logged state
```

**Step 2: Add the transition logger function**

Add near the `_append_activity` function:

```python
def _log_transitions(rows: list[dict[str, Any]]) -> None:
    """Compare current pane states against previous and log changes."""
    entries: list[dict[str, Any]] = []
    now = time.time()

    for row in rows:
        project = str(row.get("project_id") or row.get("pane_title") or "")
        if not project or project == "-":
            continue

        health = str(row.get("health") or "green")
        status = str(row.get("status") or "idle")
        agent = str(row.get("agent") or "")
        reason = str(row.get("reason") or "")
        cc = row.get("cc_stats") or {}

        current = {
            "health": health,
            "status": status,
            "agent": agent,
        }

        prev = _prev_activity_state.get(project)

        # Log on: first observation, health change, status change
        if prev is None or prev.get("health") != health or prev.get("status") != status:
            event = "state_change"
            if prev is None:
                event = "first_seen"
            elif prev.get("health") != health:
                event = f"health_{prev.get('health', '?')}_to_{health}"
            elif prev.get("status") != status:
                event = f"status_{prev.get('status', '?')}_to_{status}"

            entry: dict[str, Any] = {
                "ts": now,
                "time": datetime.now().strftime("%H:%M:%S"),
                "project": project,
                "event": event,
                "health": health,
                "status": status,
                "agent": agent,
            }
            if reason:
                entry["reason"] = reason
            if cc.get("cost") is not None:
                entry["cost"] = cc["cost"]
            if cc.get("context_pct") is not None:
                entry["context_pct"] = cc["context_pct"]

            entries.append(entry)

        _prev_activity_state[project] = current

    _append_activity(entries)
```

**Step 3: Call the transition logger from the rail loop**

In `_rail_loop` (~line 1898), after the `rows = refresh_pane_health(...)` call (~line 1908), add:

```python
        _log_transitions(rows)
```

Also log transitions from `run_watch` (~line 2152) after its `refresh_pane_health` call:

```python
            _log_transitions(rows)
```

**Step 4: Verify**

Run the rail for 10 seconds, then check the log:
```bash
timeout 12 ./scripts/agent-wrangler rail --interval 5 || true
cat .state/activity.jsonl
```

Expected: JSONL entries with `first_seen` events for each active pane.

**Step 5: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: log state transitions to activity.jsonl during rail/watch"
```

---

### Task 3: Stall detection

Distinguish between an agent that just finished (waiting for <5 min) and one that's been idle for a long time (stalled). Show stalls distinctly in the rail.

**Files:**
- Modify: `scripts/agent_wrangler.py` (add stall tracking + rail display)

**Step 1: Add stall tracking state**

Add to the module-level state variables:

```python
_last_active_time: dict[str, float] = {}  # project_id -> timestamp when last seen green/active
_STALL_THRESHOLD_MIN = 10  # minutes of waiting after being active = stalled
```

**Step 2: Update the rail loop to track and display stalls**

In `_rail_loop`, after `_log_transitions(rows)`, update the last-active tracker:

```python
        for row in rows:
            project = str(row.get("project_id") or row.get("pane_title") or "")
            if not project:
                continue
            if str(row.get("health")) == "green":
                _last_active_time[project] = time.time()
```

In the per-pane rendering section (~line 1983), after the line that builds the `line` variable, add stall indicator logic:

```python
            # Stall detection: waiting for too long after being active
            stall_label = ""
            if health == "yellow" and project in _last_active_time:
                idle_min = (time.time() - _last_active_time[project]) / 60.0
                if idle_min >= _STALL_THRESHOLD_MIN:
                    stall_label = f" \033[31m{int(idle_min)}m idle\033[0m"
                elif idle_min >= 2:
                    stall_label = f" \033[2m{int(idle_min)}m\033[0m"

            line += stall_label
```

**Step 3: Verify**

Run the rail and observe that yellow panes show idle time after a few minutes.

**Step 4: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: stall detection - show idle time for waiting agents"
```

---

### Task 4: Briefing command

`agent-wrangler briefing` reads the activity log and shows a summary. This is what you see when you come back after walking away.

**Files:**
- Modify: `scripts/agent_wrangler.py` (add `run_briefing` function + register subcommand)
- Modify: `scripts/agent-wrangler` (add route for `briefing`)

**Step 1: Add the briefing function**

Add before `register_subparser` (~line 3711):

```python
def run_briefing(args: argparse.Namespace) -> int:
    """Show what happened while you were away."""
    since = max(1, int(args.since))
    entries = _read_activity(since_minutes=since)

    if not entries:
        print(f"No activity in the last {since} minutes.")
        print(f"(Activity is recorded when the rail or watch is running.)")
        return 0

    # Group by project
    by_project: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        project = entry.get("project", "?")
        by_project.setdefault(project, []).append(entry)

    RUST = "\033[38;5;130m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    RST = "\033[0m"

    first_ts = entries[0].get("ts", 0)
    last_ts = entries[-1].get("ts", 0)
    span_min = (last_ts - first_ts) / 60.0 if first_ts and last_ts else 0

    print(f"\n  {RUST}BRIEFING{RST} — last {since} minutes ({len(entries)} events)\n")

    for project in sorted(by_project.keys()):
        events = by_project[project]
        latest = events[-1]
        health = latest.get("health", "?")
        status = latest.get("status", "?")
        agent = latest.get("agent", "")

        health_color = {"green": GREEN, "yellow": YELLOW, "red": RED}.get(health, RST)
        dot = f"{health_color}●{RST}"

        # Count transitions
        health_changes = [e for e in events if e.get("event", "").startswith("health_")]
        status_changes = [e for e in events if e.get("event", "").startswith("status_")]
        errors = [e for e in events if "red" in e.get("event", "") or e.get("health") == "red"]

        # Cost tracking
        costs = [e.get("cost") for e in events if e.get("cost") is not None]
        cost_str = ""
        if costs:
            cost_str = f"  ${costs[-1]:.2f}"
            if len(costs) > 1 and costs[-1] > costs[0]:
                cost_str += f" (+${costs[-1] - costs[0]:.2f})"

        print(f"  {dot} {project:<20} {DIM}now: {health}/{status}{RST}{cost_str}")

        # Show agent
        if agent and agent != "-":
            print(f"    {DIM}agent: {agent}{RST}")

        # Timeline of events
        for event in events:
            ev = event.get("event", "")
            t = event.get("time", "")
            reason = event.get("reason", "")
            if ev == "first_seen":
                continue
            label = ev.replace("health_", "").replace("status_", "").replace("_to_", " -> ")
            reason_str = f" ({reason})" if reason else ""
            print(f"    {DIM}{t}{RST}  {label}{reason_str}")

        if errors:
            print(f"    {RED}!! {len(errors)} error event(s){RST}")

        print()

    # Overall summary
    total_projects = len(by_project)
    current_health: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
    for events in by_project.values():
        h = events[-1].get("health", "green")
        current_health[h] = current_health.get(h, 0) + 1

    g, y, r = current_health.get("green", 0), current_health.get("yellow", 0), current_health.get("red", 0)
    print(f"  {DIM}Summary: {total_projects} projects tracked over {span_min:.0f} minutes{RST}")
    print(f"  {GREEN}{g} grazing{RST}  {YELLOW}{y} at fence{RST}  {RED}{r} down{RST}")

    # Show pane output for anything currently red or stalled
    attention_projects = [
        p for p, events in by_project.items()
        if events[-1].get("health") == "red"
    ]
    if attention_projects:
        print(f"\n  {RED}Needs attention:{RST}")
        for p in attention_projects:
            latest = by_project[p][-1]
            reason = latest.get("reason", "unknown")
            print(f"    {RED}●{RST} {p}: {reason}")
        print(f"  {DIM}Run: agent-wrangler summary <project> for details{RST}")

    print()
    return 0
```

**Step 2: Register the briefing subcommand**

In `register_subparser` (~line 3711), add after the `summary_cmd` block:

```python
    briefing_cmd = teams_sub.add_parser("briefing", help="Show what happened while you were away")
    briefing_cmd.add_argument("--since", type=int, default=60, help="Look back N minutes (default: 60)")
    briefing_cmd.set_defaults(handler=run_briefing)
```

**Step 3: Add the route in the bash router**

Read `scripts/agent-wrangler` and add `briefing` to the routing. It should route to `teams briefing` like other commands.

**Step 4: Verify**

```bash
# Run the rail briefly to generate some activity
timeout 15 ./scripts/agent-wrangler rail --interval 5 || true

# Check the briefing
./scripts/agent-wrangler briefing --since 5
```

Expected: A formatted summary showing each project's state transitions.

**Step 5: Commit**

```bash
git add scripts/agent_wrangler.py scripts/agent-wrangler
git commit -m "feat: briefing command - see what happened while you were away"
```

---

### Task 5: Auto-pilot agent launch

Ensure agents launched through agent-wrangler use `--dangerously-skip-permissions` by default. The `_graze_project` function already does this, but `run_agent` doesn't.

**Files:**
- Modify: `scripts/agent_wrangler.py` (update `run_agent` to default to auto-pilot for claude)

**Step 1: Add auto-pilot flag to agent launch**

In `run_agent` (~line 2969), after building the command tokens, inject the `--dangerously-skip-permissions` flag for Claude when not already specified:

```python
    # Auto-pilot: Claude gets --dangerously-skip-permissions unless explicitly opted out
    if tool == "claude" and "--dangerously-skip-permissions" not in " ".join(tokens):
        if not args.no_auto:
            tokens.insert(1, "--dangerously-skip-permissions")
```

**Step 2: Add the `--no-auto` flag to the agent subparser**

In `register_subparser`, in the agent subparser block (~line 3838):

```python
    agent.add_argument("--no-auto", action="store_true",
                        help="Don't add --dangerously-skip-permissions to claude")
```

**Step 3: Verify**

```bash
./scripts/agent-wrangler agent some-pane claude --help  # Won't actually run, just verify the command builds correctly
```

**Step 4: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: auto-pilot - agent launch defaults to skip-permissions for claude"
```

---

### Task 6: Persist stall data across rail restarts

The `_last_active_time` dict is in-memory only. If the rail restarts, stall context is lost. Persist it to `.state/`.

**Files:**
- Modify: `scripts/agent_wrangler.py`

**Step 1: Add persistence for last-active times**

Add near the activity log functions:

```python
_ACTIVE_TIMES_PATH = ROOT / ".state" / "active_times.json"


def _load_active_times() -> dict[str, float]:
    try:
        if _ACTIVE_TIMES_PATH.exists():
            return json.loads(_ACTIVE_TIMES_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_active_times(data: dict[str, float]) -> None:
    try:
        _ACTIVE_TIMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_TIMES_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
```

**Step 2: Load on rail start, save periodically**

At the start of `_rail_loop`, load persisted times:

```python
    global _last_active_time
    _last_active_time = _load_active_times()
```

After the stall tracking loop (where `_last_active_time` is updated), save every refresh:

```python
        _save_active_times(_last_active_time)
```

**Step 3: Verify**

Run rail, stop it, run it again. Stall times should persist.

**Step 4: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: persist stall tracking across rail restarts"
```

---

### Task 7: Log a snapshot on rail startup and shutdown

Log a "session_start" event when the rail begins and "session_end" when it stops. This makes briefing more useful — you can see when monitoring was active.

**Files:**
- Modify: `scripts/agent_wrangler.py`

**Step 1: Add session markers**

In `_rail_loop`, at the top of the function (after loading active times):

```python
    _append_activity([{
        "ts": time.time(),
        "time": datetime.now().strftime("%H:%M:%S"),
        "project": "_system",
        "event": "rail_started",
    }])
```

And in the `finally` block of `run_rail` (before `_restore_tty()`), or better, at the end of `_rail_loop` after the while loop:

After the `while True` loop ends (the `return 0` at the end of `_rail_loop`), wrap it:

```python
    _append_activity([{
        "ts": time.time(),
        "time": datetime.now().strftime("%H:%M:%S"),
        "project": "_system",
        "event": "rail_stopped",
    }])
    return 0
```

**Step 2: Update briefing to show session markers**

In `run_briefing`, add a section that shows when monitoring was active:

```python
    # Show monitoring windows
    system_events = [e for e in entries if e.get("project") == "_system"]
    if system_events:
        starts = [e for e in system_events if e.get("event") == "rail_started"]
        stops = [e for e in system_events if e.get("event") == "rail_stopped"]
        if starts:
            print(f"  {DIM}Monitoring started: {starts[-1].get('time', '?')}{RST}")
        if stops:
            print(f"  {DIM}Monitoring stopped: {stops[-1].get('time', '?')}{RST}")
        print()
```

Add this right after the "BRIEFING" header line, before the per-project loop.

**Step 3: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: log rail start/stop events for briefing context"
```

---

## Verification

After all tasks, run the full flow:

1. `./scripts/agent-wrangler start` — start the grid with agents
2. Let the rail run for 2-3 minutes
3. `./scripts/agent-wrangler briefing --since 10` — should show activity
4. Check that stall detection shows idle time on yellow panes
5. Verify `./scripts/agent-wrangler agent some-pane claude` includes `--dangerously-skip-permissions`

## Non-goals (YAGNI)

- Desktop notifications (explicitly deferred by user)
- Monolith split (separate effort, not blocking)
- Test suite (project has none; not adding now)
- Auto-recovery / auto-restart of stalled agents (future feature)
- Task queue / pre-loading tasks per project (future feature)
