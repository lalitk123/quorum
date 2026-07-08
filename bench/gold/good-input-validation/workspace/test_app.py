import pytest

from app import signup


def test_signup_accepts_valid_email():
    assert signup("a@b.com", "longenough")["status"] == "created"


def test_signup_rejects_bad_email():
    with pytest.raises(ValueError):
        signup("not-an-email", "longenough")
