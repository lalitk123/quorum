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


# ---- Authenticity vs internal consistency ------------------------------------
# The chain and the signatures both live inside the file being checked, so a
# forger who can write that file can produce a log that is perfectly
# self-consistent. Only a key obtained out-of-band distinguishes the real log
# from the forgery. These tests pin that distinction down.

def _forged_log(tmp_path):
    """A complete audit trail written by an attacker's own key."""
    log_file = str(tmp_path / "forged.jsonl")
    evil = receipts.Ed25519PrivateKey.generate()
    emit(_receipt("T-666", action="pr_opened", title="backdoor, never reviewed"),
         evil, log_file)
    return log_file, evil


def test_forged_log_is_internally_consistent(tmp_path):
    """Baseline: the forgery passes every *structural* check. This is expected —
    and precisely why an unpinned verification cannot be reported as success."""
    log_file, _ = _forged_log(tmp_path)
    results = verify_log(log_file)
    assert all(r["sig_ok"] and r["chain_ok"] for r in results)


def test_forged_log_fails_against_the_real_pinned_key(tmp_path):
    """Pinning is what actually detects the forgery."""
    log_file, _ = _forged_log(tmp_path)
    operators_key = receipts.Ed25519PrivateKey.generate()
    results = verify_log(log_file, public_key_hex(operators_key))
    assert all(not r["sig_ok"] for r in results)


def test_unpinned_report_refuses_to_claim_verified(tmp_path):
    """An unpinned check must not read as proof. It said 'N/N verified' before,
    which overstates what a self-referential signature can establish."""
    log_file, evil = _forged_log(tmp_path)
    out = receipts.format_verification(
        verify_log(log_file), observed_keys=receipts.observed_public_keys(log_file))
    assert "NOT AUTHENTICATED" in out
    summary = next(l for l in out.splitlines() if "receipt(s)" in l)
    assert "internally consistent" in summary and "verified" not in summary
    assert public_key_hex(evil) in out          # operator can see whose key signed it


def test_pinned_report_says_what_it_checked_against(tmp_path):
    log_file = str(tmp_path / "real.jsonl")
    key = receipts.Ed25519PrivateKey.generate()
    emit(_receipt("T1"), key, log_file)
    pin = public_key_hex(key)
    out = receipts.format_verification(verify_log(log_file, pin), pinned_public_key_hex=pin)
    assert "verified against pinned key" in out
    assert "NOT AUTHENTICATED" not in out


def test_observed_keys_surfaces_a_second_signer(tmp_path):
    """A log signed by two keys is either a rotation or a partial forgery; either
    way the operator must see it before pinning either key."""
    log_file = str(tmp_path / "mixed.jsonl")
    k1, k2 = receipts.Ed25519PrivateKey.generate(), receipts.Ed25519PrivateKey.generate()
    emit(_receipt("T1"), k1, log_file)
    emit(_receipt("T2"), k2, log_file)
    keys = receipts.observed_public_keys(log_file)
    assert keys == [public_key_hex(k1), public_key_hex(k2)]
    out = receipts.format_verification(verify_log(log_file), observed_keys=keys)
    assert "MORE THAN ONE SIGNER" in out


def test_receipt_records_full_panel_size_not_just_polled_seats():
    """Early exit stops polling once the outcome is forced, so `seats` can be 3 of
    a 5-seat council. A receipt reading '3 of 3 approved' would misstate a
    majority as a unanimous panel, so the true size is recorded alongside."""
    r = _receipt(approvals=3, seats=3, seats_total=5)
    record = sign_receipt(r, receipts.Ed25519PrivateKey.generate())
    assert record["payload"]["seats"] == 3
    assert record["payload"]["seats_total"] == 5
    assert verify_receipt(record)
