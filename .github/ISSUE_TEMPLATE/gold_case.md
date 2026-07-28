---
name: Gold case proposal
about: Propose a labeled benchmark case for bench/gold/
title: ''
labels: gold-case
---

**Label** — is this change `approve` (clean) or `reject` (defective)?

**The change** — paste the diff, or link a PR it is drawn from.

**Why it is interesting.** The corpus does not need more obvious cases. The
valuable ones are:

- clean changes that *look* alarming (these are what catch wolf-crier seats), or
- defects subtle enough that you would expect **two or more seats to miss them
  together** (these are what give the independence audit teeth), or
- a matched pair — the same feature written safely and unsafely.

**Which seat do you expect to miss it, and why?** A guess is fine; being wrong
about it is itself a useful result.
