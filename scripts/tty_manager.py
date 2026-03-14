#!/usr/bin/env python3
"""Direct TTY management — send commands to any terminal without tmux."""

from __future__ import annotations

import os


def tty_write(tty: str, text: str) -> bool:
    """Write text to a TTY device. Returns True on success."""
    dev = f"/dev/{tty}" if not tty.startswith("/dev/") else tty
    try:
        fd = os.open(dev, os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, text.encode())
            return True
        finally:
            os.close(fd)
    except OSError:
        return False


def tty_send_command(tty: str, command: str) -> bool:
    """Send a command + newline to a TTY."""
    return tty_write(tty, command + "\n")


def tty_send_ctrl_c(tty: str) -> bool:
    """Send Ctrl-C (interrupt) to a TTY."""
    return tty_write(tty, "\x03")


def tty_send_ctrl_d(tty: str) -> bool:
    """Send Ctrl-D (EOF) to a TTY."""
    return tty_write(tty, "\x04")
