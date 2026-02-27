# Amir Operator Agent Blueprint

## Goal

One operator agent that continuously answers:

1. What should I work on now?
2. Which terminal/repo context should be open?
3. What can be delegated to an autonomous agent?

## Architecture

```text
Linear (gabooja + amirhjalali)      Local Machine Signals
            |                                  |
            v                                  v
      Intake Collector  <----------------  Repo/Process Scanner
            |
            v
      Priority Engine (Now/Next/Later)
            |
            +--> Human Queue (today's top 3)
            |
            +--> Delegate Queue (send to linear-agent/agentcy)
            |
            +--> Hygiene Queue (close stale branches/processes)
```

## Layer 1: Intake Collector

Inputs:

- Assigned/open issues from both Linear workspaces.
- Git status per local project.
- Running process inventory by project path.
- Optional conductor status from `gabooja-agents/linear-agent`.

Output:

- Unified queue with tags:
  - `domain:business`
  - `domain:personal`
  - `type:build|ops|research|admin`
  - `mode:human|delegate`

## Layer 2: Priority Engine

Scoring dimensions:

- Urgency (Linear priority + stale age)
- Cost of context switching (active process count)
- Work in progress risk (dirty tree size)
- Domain budget limits (max 3 business + 2 personal active contexts)

Decision rule:

- Return exactly 3 "Now" items.
- Everything else goes to "Next" or "Delegate".

## Layer 3: Execution Interface

Human-mode actions:

- Launch specific context in Ghostty by project id.
- Show startup command and expected port.
- Generate quick end-of-session summary.

Delegate-mode actions:

- Assign issue to `linear-agent` bot when task is implementation-ready.
- For platform tasks, route to `agentcy` workflow.

## Layer 4: Hygiene Automation

Hourly checks:

1. Detect >5 active contexts and alert.
2. Detect repos with very large dirty trees.
3. Detect long-running servers not tied to active focus items.
4. Detect issues in "In Progress" without updates for N hours.

## Recommended rollout

### Week 1

- Use `AmirWorkflow/scripts/daily-start.sh` each morning.
- Keep project registry accurate.
- Enforce max active contexts manually.

### Week 2

- Add one automation: hourly `snapshot --include-linear`.
- Start routing low-risk coding tasks to `linear-agent`.

### Week 3

- Add queue synchronization rules between both Linear workspaces.
- Add auto-comment status updates for stale in-progress issues.

### Week 4

- Promote to true operator mode:
  - daily brief at 8:30
  - midday drift correction
  - shutdown summary with carryover plan

## Known gaps to address

1. `gabooja-agents/linear-agent` includes hardcoded `/Users/gabooja/...` defaults; switch to path config only.
2. Current setup has no strict WIP limits across business vs personal work.
3. Terminal context management is not yet session-persistent (tmux/zellij optional upgrade).

## Success criteria

- Fewer than 5 active contexts at any given time.
- Daily top 3 completed or explicitly deferred.
- No repo with >50 dirty files for more than 24 hours.
- Cross-workspace visibility from one snapshot report.
