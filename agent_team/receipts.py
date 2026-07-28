"""Signed audit receipts — a tamper-evident record of every gated action.

The gate's whole value is trust: a reviewer who cannot later prove what the
council decided, on which commit, is asking to be taken at its word. This module
emits an **Ed25519-signed receipt** for every action the team takes at the trust
boundary — a council approval, a branch push, a PR open — and appends it to a
hash-chained, append-only log. Anyone holding the public key can verify, offline
and without trusting this process, that:

- the recorded decision was produced by the holder of the signing key,
- the payload (commit SHA, council verdicts, timestamp) was not altered after
  the fact, and
- no receipt was silently removed from the middle of the log (each receipt
  carries the hash of the previous one, so a deletion breaks the chain).

Asymmetric signatures (not an HMAC) are deliberate: a third-party auditor in a
regulated context can verify receipts with the public key alone, while the
secret never leaves the operator. Everything here is deterministic and offline —
no model, no network.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

log = logging.getLogger("receipts")

RECEIPT_VERSION = 1
GENESIS_PREV = "0" * 64   # prev_hash of the first receipt in a chain


@dataclass
class Receipt:
    """The signed payload. Field order is irrelevant — signing canonicalizes."""
    action: str                        # approved | pr_opened | ...
    ticket_id: str
    title: str
    branch: str
    base: str
    commit_sha: str
    timestamp: str                     # ISO-8601, supplied by the caller
    council_rule: str
    approvals: int
    seats: int                         # seats POLLED (early exit may stop short)
    verdicts: list[dict] = field(default_factory=list)  # [{reviewer, approve}]
    seats_total: int = 0               # full panel size; 0 = not recorded (legacy)
    risk_level: str = ""
    merge_disposition: str = ""
    trust_score: float = 0.0
    trust_grade: str = ""
    pr_url: str = ""
    prev_hash: str = GENESIS_PREV
    version: int = RECEIPT_VERSION


def _canonical(payload: dict) -> bytes:
    """Deterministic bytes for signing/hashing — sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def receipt_hash(record: dict) -> str:
    """Hash of a full signed record (payload + signature) — the chain link."""
    return sha256(_canonical(record)).hexdigest()


# ---- Key management ----------------------------------------------------------

def load_or_create_key(key_file: str) -> Ed25519PrivateKey:
    """Load an Ed25519 private key (PEM), creating and persisting one if absent.

    A generated key is logged loudly: an audit trail is only meaningful against a
    stable, operator-controlled key, so ephemeral keys are a misconfiguration to
    surface, not to hide.
    """
    path = Path(key_file)
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    key = Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    try:
        path.chmod(0o600)
    except OSError:
        pass
    log.warning("generated a NEW audit signing key at %s — back it up and pin its "
                "public key; receipts before this key existed cannot be verified "
                "against it", key_file)
    return key


def public_key_hex(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


# ---- Sign / verify -----------------------------------------------------------

def sign_receipt(receipt: Receipt, key: Ed25519PrivateKey) -> dict:
    payload = asdict(receipt)
    signature = key.sign(_canonical(payload))
    return {
        "payload": payload,
        "alg": "ed25519",
        "public_key": public_key_hex(key),
        "signature": signature.hex(),
    }


def verify_receipt(record: dict, pinned_public_key_hex: str | None = None) -> bool:
    """Verify one signed record's signature (and an optional pinned key).

    `pinned_public_key_hex` guards against a forged receipt that ships its own
    attacker-controlled key: when given, the record's embedded key must match it.
    """
    try:
        embedded = record["public_key"]
        if pinned_public_key_hex and embedded != pinned_public_key_hex:
            return False
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(embedded))
        pub.verify(bytes.fromhex(record["signature"]), _canonical(record["payload"]))
        return True
    except (KeyError, ValueError, InvalidSignature):
        return False


# ---- Append-only, hash-chained log ------------------------------------------

def last_hash(receipts_file: str) -> str:
    """Hash of the final receipt in the log (the next receipt's prev_hash)."""
    path = Path(receipts_file)
    if not path.exists():
        return GENESIS_PREV
    last = ""
    for line in path.read_text().splitlines():
        if line.strip():
            last = line
    if not last:
        return GENESIS_PREV
    return receipt_hash(json.loads(last))


def append_receipt(receipts_file: str, record: dict) -> None:
    path = Path(receipts_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def emit(receipt: Receipt, key: Ed25519PrivateKey, receipts_file: str) -> dict:
    """Chain a receipt onto the log's tip, sign it, and append it."""
    receipt.prev_hash = last_hash(receipts_file)
    record = sign_receipt(receipt, key)
    append_receipt(receipts_file, record)
    return record


def verify_log(receipts_file: str, pinned_public_key_hex: str | None = None) -> list[dict]:
    """Verify every receipt's signature and the prev_hash chain end to end.

    Returns one result dict per receipt: {index, ticket_id, action, sig_ok,
    chain_ok}. A middle-of-log deletion breaks `chain_ok` at the next receipt
    even though every signature is individually valid.
    """
    path = Path(receipts_file)
    results: list[dict] = []
    if not path.exists():
        return results
    prev = GENESIS_PREV
    for i, line in enumerate(l for l in path.read_text().splitlines() if l.strip()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            results.append({"index": i, "ticket_id": "?", "action": "?",
                            "sig_ok": False, "chain_ok": False})
            continue
        sig_ok = verify_receipt(record, pinned_public_key_hex)
        payload = record.get("payload", {})
        chain_ok = payload.get("prev_hash") == prev
        results.append({
            "index": i,
            "ticket_id": payload.get("ticket_id", "?"),
            "action": payload.get("action", "?"),
            "sig_ok": sig_ok,
            "chain_ok": chain_ok,
        })
        prev = receipt_hash(record)
    return results


def observed_public_keys(receipts_file: str) -> list[str]:
    """Every distinct signing key appearing in the log, in first-seen order.

    More than one means the key rotated — or that part of the log was written by
    a different signer, which is what a wholesale forgery looks like.
    """
    path = Path(receipts_file)
    if not path.exists():
        return []
    seen: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            key = json.loads(line).get("public_key")
        except json.JSONDecodeError:
            continue
        if key and key not in seen:
            seen.append(key)
    return seen


def format_verification(results: list[dict], pinned_public_key_hex: str | None = None,
                        observed_keys: list[str] | None = None) -> str:
    """Render the verification report.

    Without a pinned key this reports *internal consistency only*, and says so.
    Signatures are checked against a key carried inside each record, so a forger
    who can write the file can re-sign every receipt with their own key and
    produce a log that is perfectly self-consistent. Authenticity requires
    comparing against a key obtained out-of-band — that is what pinning is.
    """
    if not results:
        return "no receipts found."
    lines = ["", "=== Audit receipt verification ==="]
    ok = 0
    for r in results:
        good = r["sig_ok"] and r["chain_ok"]
        ok += good
        flags = []
        if not r["sig_ok"]:
            flags.append("BAD SIGNATURE")
        if not r["chain_ok"]:
            flags.append("BROKEN CHAIN")
        mark = "ok  " if good else "FAIL"
        lines.append(f"  [{mark}] #{r['index']:<3} {r['action']:<10} {r['ticket_id']:<16}"
                     + (f"  <- {', '.join(flags)}" if flags else ""))

    consistent = f"{ok}/{len(results)} receipt(s)"
    if pinned_public_key_hex:
        lines.append(f"{consistent} verified against pinned key "
                     f"{pinned_public_key_hex[:16]}…"
                     + ("" if ok == len(results) else "  — AUDIT TRAIL COMPROMISED"))
        return "\n".join(lines)

    lines.append(f"{consistent} internally consistent"
                 + ("" if ok == len(results) else "  — AUDIT TRAIL COMPROMISED"))
    lines.append("")
    lines.append("  NOT AUTHENTICATED — no public key was pinned.")
    lines.append("  Each signature was checked against a key carried inside the record "
                 "itself, so")
    lines.append("  this proves the log is self-consistent, NOT that your council "
                 "produced it. Anyone")
    lines.append("  who can write this file can re-sign every entry with their own key "
                 "and pass.")
    for k in observed_keys or []:
        lines.append(f"    key observed in log: {k}")
    if observed_keys and len(observed_keys) > 1:
        lines.append("    ^ MORE THAN ONE SIGNER — investigate before pinning either.")
    lines.append("  Confirm the key out-of-band, then set audit.public_key in "
                 "config.yaml to pin it.")
    return "\n".join(lines)
