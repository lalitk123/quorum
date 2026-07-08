"""Backlog sources: a local tasks.yaml or open GitHub issues via `gh`."""

import json
import subprocess

import yaml

from .models import Ticket


def load_backlog(cfg: dict) -> list[Ticket]:
    source = cfg["backlog"]["source"]
    if source == "file":
        return _from_file(cfg["backlog"]["file"])
    if source == "github":
        return _from_github(cfg["backlog"]["github_repo"])
    raise ValueError(f"unknown backlog source: {source}")


def _from_file(path: str) -> list[Ticket]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        Ticket(id=str(t["id"]), title=t["title"], description=t.get("description", ""),
               depends_on=[str(d) for d in t.get("depends_on", [])])
        for t in data["tasks"]
    ]


def _from_github(repo: str) -> list[Ticket]:
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--state", "open",
         "--json", "number,title,body"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [
        Ticket(id=f"gh-{i['number']}", title=i["title"], description=i.get("body") or "")
        for i in json.loads(out)
    ]
