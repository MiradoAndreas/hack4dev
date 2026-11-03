"""Check project-local skill discovery prerequisites without third-party packages.

This is a static metadata check, not proof that a running agent has refreshed
its skill catalog, nor a replacement for reviewing bundled upstream resources.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_skill(name: str) -> bool:
    path = PROJECT_ROOT / ".agents" / "skills" / name / "SKILL.md"
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"MISSING/UNREADABLE: {name}: {exc}", file=sys.stderr)
        return False
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", content, re.DOTALL)
    if match is None:
        print(f"INVALID: {name}: missing YAML frontmatter", file=sys.stderr)
        return False
    metadata = match.group(1)
    declared_name = re.search(r"^name:\s*([^\r\n]+)$", metadata, re.MULTILINE)
    description = re.search(r"^description:\s*(\S[\s\S]*)", metadata, re.MULTILINE)
    if declared_name is None or declared_name.group(1).strip().strip("\"'") != name:
        print(f"INVALID: {name}: frontmatter name does not match directory", file=sys.stderr)
        return False
    if description is None:
        print(f"INVALID: {name}: missing description", file=sys.stderr)
        return False
    if not content[match.end():].strip():
        print(f"INVALID: {name}: empty skill instructions", file=sys.stderr)
        return False
    print(f"READY FOR DISCOVERY: {name}: {path.relative_to(PROJECT_ROOT)}")
    return True


if __name__ == "__main__":
    if not sys.argv[1:]:
        sys.exit("Usage: python3 scripts/verify-agent-skills.py SKILL [SKILL ...]")
    results = [verify_skill(name) for name in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
