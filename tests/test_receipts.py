"""Signed audit-receipt tests: sign/verify roundtrip, tamper detection, and the
hash-chain that catches a deleted receipt. Deterministic and offline.
"""

import json

import pytest

from agent_team import receipts
from agent_team.receipts import (
    GENESIS_PREV, Receipt, emit, load_or_create_key, public_key_hex,
    sign_receipt, verify_log, verify_receipt,
)


def _receipt(ticket_id="T1", action="approved", **kw):
    base = dict(
        action=action, ticket_id=ticket_id, title="do a thing",
        branch=f"agent/{ticket_id}", base="main", commit_sha="deadbeef",
        timestamp="2026-07-08T00:00:00+00:00", council_rule="unanimous",
        approvals=5, seats=5,
        verdicts=[{"reviewer": "security", "approve": True}],
    )
    base.update(kw)
    return Receipt(**base)


def test_sign_and_verify_roundtrip():
    key = receipts.Ed25519PrivateKey.generate()
    record = sign_receipt(_receipt(), key)
    assert verify_receipt(record) is True
    assert record["public_key"] == public_key_hex(key)


def test_tampered_payload_fails_verification():
    key = receipts.Ed25519PrivateKey.generate()
    record = sign_receipt(_receipt(commit_sha="deadbeef"), key)
    record["payload"]["commit_sha"] = "f00dface"   # forge the signed commit
    assert verify_receipt(record) is False


def test_pinned_public_key_rejects_foreign_key():
    honest = receipts.Ed25519PrivateKey.generate()
    attacker = receipts.Ed25519PrivateKey.generate()
    # A receipt validly signed by the attacker, shipping the attacker's own key.
    forged = sign_receipt(_receipt(), attacker)
    assert verify_receipt(forged) is True                       # self-consistent
    assert verify_receipt(forged, public_key_hex(honest)) is False  # not the pinned key


def test_key_is_persisted_and_reloaded(tmp_path):
    key_file = str(tmp_path / "audit_key.pem")
    k1 = load_or_create_key(key_file)
    k2 = load_or_create_key(key_file)          # loads, does not regenerate
    assert public_key_hex(k1) == public_key_hex(k2)


def test_emit_chains_receipts_and_verifies(tmp_path):
    key_file = str(tmp_path / "k.pem")
    log_file = str(tmp_path / "receipts.jsonl")
    key = load_or_create_key(key_file)

    emit(_receipt("T1"), key, log_file)
    emit(_receipt("T2"), key, log_file)
    emit(_receipt("T3"), key, log_file)

    lines = [json.loads(l) for l in open(log_file)]
    assert lines[0]["payload"]["prev_hash"] == GENESIS_PREV
    assert lines[1]["payload"]["prev_hash"] == receipts.receipt_hash(lines[0])

    results = verify_log(log_file, public_key_hex(key))
    assert len(results) == 3
    assert all(r["sig_ok"] and r["chain_ok"] for r in results)


def test_deleting_a_middle_receipt_breaks_the_chain(tmp_path):
    key_file = str(tmp_path / "k.pem")
    log_file = str(tmp_path / "receipts.jsonl")
    key = load_or_create_key(key_file)
    for tid in ("T1", "T2", "T3"):
        emit(_receipt(tid), key, log_file)

    lines = open(log_file).read().splitlines()
    del lines[1]                                # excise the middle receipt
    with open(log_file, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    results = verify_log(log_file, public_key_hex(key))
    assert len(results) == 2
    assert all(r["sig_ok"] for r in results)    # each remaining signature is valid
    assert results[0]["chain_ok"] is True       # T1 still links to genesis
    assert results[1]["chain_ok"] is False      # T3's prev_hash no longer matches


def test_verify_log_missing_file_is_empty(tmp_path):
    assert verify_log(str(tmp_path / "nope.jsonl")) == []
