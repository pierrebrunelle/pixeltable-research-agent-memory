"""Pixeltable UDFs and UDAs for the research-assistant agent memory (recorded as `udfs.<name>`).

Pure Python, no table imports. app.py imports these; tables and @pxt.query live in app.py.
"""

from __future__ import annotations

import datetime as dt
import re

import pixeltable as pxt

CHAT_MODEL = 'gpt-4o-mini'
EMBED_MODEL = 'text-embedding-3-small'

_REMEMBER_RE = re.compile(r'^\s*(?:please\s+)?remember(?:\s+that)?\s+(?P<fact>.+?)\s*[.!]?\s*$', re.I)
_PREFER_RE = re.compile(r'\bI\s+(?:prefer|like|always want|never want)\b.+', re.I)


@pxt.udf
def extract_memory(message: str) -> pxt.Json | None:
    """Rule-based write-back: 'remember that ...' -> a fact, 'I prefer ...' -> a preference, else None."""
    m = _REMEMBER_RE.match(message or '')
    if m:
        return {'kind': 'fact', 'content': m.group('fact')[0].upper() + m.group('fact')[1:], 'importance': 4}
    p = _PREFER_RE.search(message or '')
    if p:
        return {'kind': 'preference', 'content': p.group(0).rstrip('.') , 'importance': 3}
    return None


@pxt.udf
def reading_tier(year: int) -> str:
    """Classify a publication year for the reading list: 'background' before 2018, else 'current'."""
    return 'background' if year < 2018 else 'current'


@pxt.udf
def memory_block(memories: pxt.Json) -> str:
    """Format recalled memories ([{kind, content, score}, ...]) as a bullet list for the prompt."""
    if not memories:
        return '(no relevant memories)'
    return '\n'.join(f'- [{m.get("kind")}] {m.get("content")}' for m in memories)


@pxt.udf
def agent_messages(user_id: str, message: str, memory_text: str) -> pxt.Json:
    """First-pass chat messages: recalled memories + the user's message; tools may be called."""
    return [
        {
            'role': 'system',
            'content': f'You are a research assistant for user "{user_id}". Use the memories when relevant. '
            'Call lookup_source when the user asks about a specific source id (like S-02), '
            'and reading_tier to classify a publication year. '
            'Answer in at most two sentences.',
        },
        {'role': 'user', 'content': f'Memories:\n{memory_text}\n\nMessage: {message}'},
    ]


@pxt.udf
def answer_messages(message: str, memory_text: str, tool_output: pxt.Json) -> pxt.Json:
    """Second-pass messages: the same context plus whatever the tools returned."""
    return [
        {'role': 'system', 'content': 'You are a research assistant. Answer in at most two sentences.'},
        {'role': 'user', 'content': f'Memories:\n{memory_text}\n\nTool results: {tool_output}\n\nMessage: {message}'},
    ]


@pxt.udf
def pick_answer(first_text: str | None, final_text: str | None, tool_output: pxt.Json) -> str:
    """Use the post-tool answer when a tool ran, else the first answer."""
    ran_tool = any(v for v in (tool_output or {}).values())
    return (final_text if ran_tool else first_text) or final_text or first_text or ''


@pxt.uda
class salience(pxt.Aggregator):
    """Importance with exponential recency decay: sum(importance * 0.5 ** (age_days / half_life)).

    Age is measured against the newest memory in the group, so the result is deterministic.
    """

    def __init__(self, half_life_days: int = 30):
        self.half_life = float(half_life_days or 30)
        self.items: list[tuple[int, dt.datetime]] = []

    def update(self, importance: int, created_at: dt.datetime) -> None:
        if importance is not None and created_at is not None:
            self.items.append((importance, created_at))

    def value(self) -> float:
        if not self.items:
            return 0.0
        newest = max(ts for _, ts in self.items)
        total = sum(imp * 0.5 ** ((newest - ts).total_seconds() / 86400.0 / self.half_life) for imp, ts in self.items)
        return round(total, 3)


@pxt.uda
class memory_digest(pxt.Aggregator):
    """Consolidate a user's memories: counts per kind plus the top-3 by importance (newest wins ties)."""

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self.by_kind: dict[str, int] = {}
        self.items: list[tuple[int, dt.datetime, str]] = []

    def update(self, kind: str, content: str, importance: int, created_at: dt.datetime) -> None:
        if kind is None or content is None:
            return
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if kind != 'summary':
            self.items.append((importance or 0, created_at, content))

    def value(self) -> pxt.Json:
        top = sorted(self.items, key=lambda t: (t[0], t[1]), reverse=True)[: self.top_n]
        return {
            'n': sum(self.by_kind.values()),
            'by_kind': dict(sorted(self.by_kind.items())),
            'top': [c for _, _, c in top],
        }


@pxt.udf
def digest_text(digest: pxt.Json) -> str:
    """One-line consolidated memory from a memory_digest result."""
    top = '; '.join((digest or {}).get('top') or [])
    return f'Profile summary ({(digest or {}).get("n", 0)} memories): {top}'
