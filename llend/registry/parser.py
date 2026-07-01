"""Skill input parser — converts YAML frontmatter ``inputs:`` strings into typed dicts.

Spec reference
==============
- **§2.4** → parsing rules, regex pattern, edge-case handling
"""

from __future__ import annotations


def parse_inputs(raw: str) -> dict[str, str]:
    """Parse ``inputs: name:type=default, name:type`` into ``{name: type_spec}``.

    Spec 002 §2.4 — the canonical input-type syntax parser used by
    ``SkillRegistry.discover()`` when reading SKILL.md frontmatter.

    Parsing rules (from §2.4 table):

    +----------------------+-----------------------------+-------------------------------------+
    | Rule                 | Input                       | Output                              |
    +======================+=============================+=====================================+
    | Split on ``,``        | ``a:str, b:int=5``          | ``["a:str", "b:int=5"]``            |
    +----------------------+-----------------------------+-------------------------------------+
    | Split each on first   | ``a:str``                   | key=``a``, type=``str``             |
    | ``:``                 |                             |                                     |
    +----------------------+-----------------------------+-------------------------------------+
    | Split type on ``=``   | ``int=5``                   | type=``int``, default=``5``         |
    +----------------------+-----------------------------+-------------------------------------+
    | Quoted defaults       | ``str="hello, world"``      | key=``msg``, type=``str``,          |
    | preserved             |                             | default=``"hello, world"``          |
    +----------------------+-----------------------------+-------------------------------------+
    | Trailing whitespace   | ``a:str , b:int``           | ``{"a": "str", "b": "int"}``        |
    | trimmed               |                             |                                     |
    +----------------------+-----------------------------+-------------------------------------+

    Commas inside double-quoted strings and square brackets are preserved
    (not treated as delimiters).  Empty or whitespace-only input returns ``{}``.
    """
    if not raw.strip():
        return {}

    # Split on top-level commas only (not inside quotes or brackets)
    parts = _split_top_level(raw, ",")

    result: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Parts without ":" are skipped — §2.4 parsing rules
        if ":" not in part:
            continue

        # Split on first colon only
        key, _, type_spec = part.partition(":")
        key = key.strip()
        if not key:
            continue

        result[key] = type_spec.strip()

    return result


# ---------------------------------------------------------------------------
# Internal — top-level split (respects quotes and brackets)
# ---------------------------------------------------------------------------


def _split_top_level(text: str, delimiter: str) -> list[str]:
    """Split *text* on *delimiter*, but only when not inside ``"..."`` or ``[...]``."""
    parts: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    in_quotes = False

    for ch in text:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "[" and not in_quotes:
            bracket_depth += 1
        elif ch == "]" and not in_quotes:
            bracket_depth -= 1
        elif ch == delimiter and bracket_depth == 0 and not in_quotes:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)

    if current:
        parts.append("".join(current))

    return parts
