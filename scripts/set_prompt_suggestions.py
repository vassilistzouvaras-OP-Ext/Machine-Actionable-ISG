#!/usr/bin/env python3
"""
Replace Open WebUI's default "new chat" starter prompt chips (the stock
ChatGPT-style examples) with custom ones, via the admin API.

Usage:
  python scripts/set_prompt_suggestions.py --deployment ISGBot
  python scripts/set_prompt_suggestions.py --url http://127.0.0.1:8081 --email a@b.com --password secret
  python scripts/set_prompt_suggestions.py --deployment ISGBot --suggestions-file my_suggestions.json

With --deployment, the Open WebUI URL and admin credentials are read from
.deployments/<name>/.env (written by deploy.py). Pass --url/--email/--password
to target a deployment that wasn't created with deploy.py.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENTS_DIR = PROJECT_ROOT / ".deployments"

DEFAULT_SUGGESTIONS: List[Dict[str, object]] = [
    {
        "title": ["Bullet points", "How should bullet points be formatted in a legislative text?"],
        "content": "How should bullet points be formatted in a legislative text?",
    },
    {
        "title": ["Footnote references", "Can the reference to a footnote be composed in italics?"],
        "content": "Can the reference to a footnote be composed in italics?",
    },
    {
        "title": ["Member States", "Can member states be listed in English alphabetical order in a table?"],
        "content": "Can member states be listed in English alphabetical order in a table?",
    },
]


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        values[key.strip()] = raw_value.replace("\\n", "\n").replace("\\\\", "\\")
    return values


def http_json(url: str, payload: dict, token: str = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Request to {url} failed ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc}")


def resolve_target(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.url and args.email and args.password:
        return args.url.rstrip("/"), args.email, args.password

    if not args.deployment:
        raise SystemExit("Pass --deployment <name>, or all three of --url/--email/--password.")

    env_path = DEPLOYMENTS_DIR / args.deployment / ".env"
    env = parse_env_file(env_path)
    if not env:
        raise SystemExit(f"No deployment env file found at {env_path}.")

    url = args.url or env.get("OPENWEBUI_URL") or env.get("REMOTE_OPENWEBUI_URL")
    email = args.email or env.get("WEBUI_ADMIN_EMAIL")
    password = args.password or env.get("WEBUI_ADMIN_PASSWORD")
    missing = [n for n, v in (("URL", url), ("admin email", email), ("admin password", password)) if not v]
    if missing:
        raise SystemExit(f"Could not resolve {', '.join(missing)} from {env_path}.")
    return url.rstrip("/"), email, password


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", help="Deployment name under .deployments/ to read URL + admin creds from")
    parser.add_argument("--url", help="Open WebUI base URL, e.g. http://127.0.0.1:8081")
    parser.add_argument("--email", help="Admin email (overrides the deployment .env)")
    parser.add_argument("--password", help="Admin password (overrides the deployment .env)")
    parser.add_argument(
        "--suggestions-file",
        type=Path,
        help='JSON file with a list of {"title": [..], "content": ".."} objects',
    )
    args = parser.parse_args()

    url, email, password = resolve_target(args)

    suggestions = DEFAULT_SUGGESTIONS
    if args.suggestions_file:
        suggestions = json.loads(args.suggestions_file.read_text(encoding="utf-8"))

    print(f"Signing in to {url} as {email} ...")
    signin = http_json(f"{url}/api/v1/auths/signin", {"email": email, "password": password})
    token = signin.get("token")
    if not token:
        raise SystemExit(f"Sign-in did not return a token: {signin}")

    print(f"Setting {len(suggestions)} default prompt suggestion(s) ...")
    result = http_json(f"{url}/api/v1/configs/suggestions", {"suggestions": suggestions}, token=token)
    for item in result if isinstance(result, list) else []:
        print(f"  - {item.get('content')}")
    print("Done. Refresh Open WebUI's new-chat screen to see the new starter prompts.")


if __name__ == "__main__":
    main()
