import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> None:
    if not _EMAIL_RE.match(email):
        raise ValueError("invalid email format")


def signup(email: str, password: str) -> dict:
    validate_email(email)
    if len(password) < 8:
        raise ValueError("password too short")
    return {"email": email, "status": "created"}
