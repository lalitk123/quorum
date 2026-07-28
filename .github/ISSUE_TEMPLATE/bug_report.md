---
name: Bug report
about: Something behaved incorrectly
title: ''
labels: bug
---

**What happened, and what did you expect instead?**

**Steps to reproduce**

**Config** — the relevant part of your `config.yaml` (seats, gates, budgets).
Please redact keys; only env-var *names* should ever appear.

**Environment** — Python version, OS, `pip show claude-agent-sdk`.

**Did the gate fail open?** If a change was approved that should have been
rejected, say so here — that is the most serious class of bug in this project
and will be prioritized accordingly.
