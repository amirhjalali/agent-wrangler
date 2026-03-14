#!/usr/bin/env python3
"""Optional Ghostty integration via AppleScript (macOS only)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def _osascript(script: str, timeout: int = 10) -> tuple[bool, str]:
    """Run AppleScript. Returns (success, output)."""
    if sys.platform != "darwin":
        return False, "not macOS"
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode == 0, proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def is_available() -> bool:
    """Check if Ghostty is running and scriptable."""
    ok, out = _osascript(
        'tell application "System Events" to (name of processes) contains "Ghostty"'
    )
    return ok and out == "true"


def list_tabs() -> list[dict[str, Any]]:
    """List all Ghostty tabs with their working directories."""
    script = '''
tell application "Ghostty"
    set results to {}
    repeat with w in windows
        repeat with t in tabs of w
            set term to focused terminal of t
            set tid to id of term
            set tname to name of t
            set tdir to working directory of term
            set tidx to index of t
            set end of results to (tid & "||" & tname & "||" & tdir & "||" & tidx)
        end repeat
    end repeat
    return results
end tell
'''
    ok, output = _osascript(script)
    if not ok or not output:
        return []
    tabs = []
    for line in output.split(", "):
        parts = line.strip().split("||")
        if len(parts) >= 4:
            tabs.append({
                "terminal_id": parts[0],
                "name": parts[1],
                "working_directory": parts[2],
                "index": parts[3],
            })
    return tabs


def create_tab(working_directory: str, command: str | None = None) -> bool:
    """Create a new Ghostty tab with the given working directory."""
    cmd_part = ""
    if command:
        cmd_part = f'\n        set command of cfg to "{command}"'
    script = f'''
tell application "Ghostty"
    set cfg to new surface configuration
    set initial working directory of cfg to "{working_directory}"{cmd_part}
    new tab in front window with configuration cfg
end tell
'''
    ok, _ = _osascript(script)
    return ok


def focus_tab(terminal_id: str) -> bool:
    """Focus a specific terminal by ID."""
    script = f'''
tell application "Ghostty"
    repeat with w in windows
        repeat with t in tabs of w
            set term to focused terminal of t
            if id of term is "{terminal_id}" then
                select t
                focus term
                return true
            end if
        end repeat
    end repeat
    return false
end tell
'''
    ok, _ = _osascript(script)
    return ok


def set_tab_title(terminal_id: str, title: str) -> bool:
    """Set the title of a specific tab."""
    script = f'''
tell application "Ghostty"
    repeat with w in windows
        repeat with t in tabs of w
            set term to focused terminal of t
            if id of term is "{terminal_id}" then
                perform action "set_tab_title:{title}" on term
                return true
            end if
        end repeat
    end repeat
    return false
end tell
'''
    ok, _ = _osascript(script)
    return ok


def send_text(terminal_id: str, text: str) -> bool:
    """Send text input to a specific terminal."""
    escaped = text.replace('\\', '\\\\').replace('"', '\\"')
    script = f'''
tell application "Ghostty"
    repeat with w in windows
        repeat with t in tabs of w
            set term to focused terminal of t
            if id of term is "{terminal_id}" then
                input text "{escaped}" to term
                return true
            end if
        end repeat
    end repeat
    return false
end tell
'''
    ok, _ = _osascript(script)
    return ok


def close_tab(terminal_id: str) -> bool:
    """Close a specific tab by terminal ID."""
    script = f'''
tell application "Ghostty"
    repeat with w in windows
        repeat with t in tabs of w
            set term to focused terminal of t
            if id of term is "{terminal_id}" then
                close t
                return true
            end if
        end repeat
    end repeat
    return false
end tell
'''
    ok, _ = _osascript(script)
    return ok
