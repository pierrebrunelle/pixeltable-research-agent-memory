"""Research Agent Memory: durable, searchable memory for an AI research assistant, in Pixeltable tables.

Sources are the papers and pages the assistant has read (with an optional cover image). Memories are
facts, preferences and episodes per user, embedded with OpenAI text-embedding-3-small into an
EmbeddingIndex. recall_memories (@pxt.query) is semantic recall scoped to one user; lookup_source
(@pxt.query) is an exact-key lookup the LLM can call as a tool. Turns is the agent table: each
message recalls memories, calls gpt-4o-mini with tools, and proposes a memory to write back.
run.py also builds pxt.retrieval_udf lookups on the live tables and consolidates memories with the
salience and memory_digest aggregates from udfs.py.

Layout: UDFs and UDAs live in udfs.py; this file holds table models and @pxt.query functions.
"""

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import openai

from udfs import (
    CHAT_MODEL,
    EMBED_MODEL,
    agent_messages,
    answer_messages,
    extract_memory,
    memory_block,
    pick_answer,
    reading_tier,
)

TableModel = pxt.model_base()


class Sources(TableModel, name='sources', has_default_idxs=False):
    """A paper, web page or note the assistant has read."""

    source_id = pxt.Column(type=pxt.String, primary_key=True)
    title: pxt.String
    kind: pxt.String
    year: pxt.Int
    url: pxt.String | None
    cover: pxt.Image | None

    cover_width = cover.width  # type: ignore[attr-defined]


class Memories(TableModel, name='memories', has_default_idxs=False):
    """Long-term memory: one fact, preference, episode or summary per row, per user."""

    memory_id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    user_id: pxt.String
    kind: pxt.String
    content: pxt.String
    importance: pxt.Int
    created_at: pxt.Timestamp
    source_id: pxt.String | None

    __indexes__ = [
        pxt.EmbeddingIndex(
            content,  # type: ignore[name-defined]  # noqa: F821
            string_embed=openai.embeddings.using(model=EMBED_MODEL),
            metric='cosine',
            name='memory_embed',
        ),
        pxt.BtreeIndex(user_id),  # type: ignore[name-defined]  # noqa: F821
    ]


@pxt.query
def recall_memories(user_id: str, context: str, k: int = 4):
    """Top-k memories of one user that are semantically closest to the context."""
    sim = Memories.content.similarity(string=context, idx='memory_embed')
    return (
        Memories.where(Memories.user_id == user_id)
        .order_by(sim, asc=False)
        .limit(k)
        .select(Memories.kind, Memories.content, Memories.source_id, score=sim)
    )


@pxt.query
def lookup_source(source_id: str):
    """Look up one source (title, kind, year, url) by its id, e.g. S-02."""
    return Sources.where(Sources.source_id == source_id).limit(1).select(
        Sources.source_id, Sources.title, Sources.kind, Sources.year, Sources.url
    )


@pxt.query
def recent_memories(user_id: str, n: int = 5):
    """The newest n memories of one user."""
    return (
        Memories.where(Memories.user_id == user_id)
        .order_by(Memories.created_at, asc=False)
        .limit(n)
        .select(Memories.kind, Memories.content, Memories.created_at)
    )


# Keep at least one @pxt.udf in the toolset: with only @pxt.query tools, invoke_tools waits
# forever on a response that has no tool calls (seen on pixeltable 0.7.16).
TOOLS = pxt.tools(lookup_source, reading_tier)


class Turns(TableModel, name='turns', has_default_idxs=False):
    """Agent table: message -> recall -> tool-calling LLM -> answer + a proposed memory write-back."""

    turn_id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    user_id: pxt.String
    message: pxt.String

    recalled = recall_memories(user_id, message, 4)
    memory_text = memory_block(recalled)
    response = openai.chat_completions(
        agent_messages(user_id, message, memory_text),
        model=CHAT_MODEL,
        model_kwargs={'max_tokens': 150, 'temperature': 0},
        tools=TOOLS,
    )
    tool_output = openai.invoke_tools(TOOLS, response)
    final = openai.chat_completions(
        answer_messages(message, memory_text, tool_output),
        model=CHAT_MODEL,
        model_kwargs={'max_tokens': 160, 'temperature': 0},
    )
    answer = pick_answer(response.choices[0].message.content, final.choices[0].message.content, tool_output)
    remember = extract_memory(message)
