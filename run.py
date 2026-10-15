#!/usr/bin/env python3
"""Research Agent Memory driver (rag-agent shape, no HTTP router).

1. pxt schema update app.py <catalog>   (sources, memories + embedding index, agent turns)
2. Load sources (with cover images) and seed memories for two users
3. Semantic recall with recall_memories, scoped per user
4. pxt.retrieval_udf lookups built on the live tables: enrich memories with their source,
   and list memories by (user_id, kind)
5. Agent turns: recall -> gpt-4o-mini with a lookup tool -> answer -> memory write-back
6. A retrieval_udf as an LLM tool in an ad-hoc query
7. Consolidation with UDAs: salience (recency-decayed importance) and memory_digest per user
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pixeltable as pxt
from pixeltable.functions import openai

from udfs import CHAT_MODEL, digest_text, memory_digest, reading_tier, salience

ROOT = Path(__file__).resolve().parent
CATALOG = os.environ.get('MEMORY_CATALOG', 'research')
PXT = str(Path(sys.executable).with_name('pxt'))
LOG = ROOT / 'logs' / 'run.jsonl'


def log(event: str, **payload) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(json.dumps({'event': event, **payload}, default=str) + '\n')
    print(f'{event}: {json.dumps(payload, default=str)[:700]}', flush=True)


def rows(rs) -> list[dict]:
    return [dict(r) for r in rs]


def main() -> int:
    if LOG.exists():
        LOG.unlink()
    t0 = time.time()
    r = subprocess.run([PXT, 'schema', 'update', 'app.py', CATALOG, '-f'], cwd=str(ROOT), capture_output=True, text=True)
    log('schema_update', exit=r.returncode, seconds=round(time.time() - t0, 2), out=r.stdout.strip()[-300:], err=r.stderr.strip()[-300:])
    if r.returncode != 0:
        return 1

    sources = pxt.get_table(f'{CATALOG}/sources')
    memories = pxt.get_table(f'{CATALOG}/memories')
    turns = pxt.get_table(f'{CATALOG}/turns')

    # --- 2. seed ---
    if sources.count() == 0:
        data = json.loads((ROOT / 'data' / 'sources.json').read_text())
        for s in data:
            if s['cover']:
                s['cover'] = str(ROOT / s['cover'])
        st = sources.insert(data)
        log('sources_loaded', n=sources.count(), status=str(st))
    if memories.count() == 0:
        seed = json.loads((ROOT / 'data' / 'memories.json').read_text())
        for m in seed:
            m['created_at'] = dt.datetime.fromisoformat(m['created_at'])
        t0 = time.time()
        st = memories.insert(seed)
        log('memories_seeded', n=memories.count(), seconds=round(time.time() - t0, 2), status=str(st))
    log('covers', rows=rows(sources.select(sources.source_id, sources.cover_width).order_by(sources.source_id).collect()))

    # --- 3. semantic recall per user ---
    q = 'how do agents decide which memories matter?'
    sim = memories.content.similarity(string=q, idx='memory_embed')
    hits = rows(
        memories.where(memories.user_id == 'ada').order_by(sim, asc=False).limit(3)
        .select(memories.content, score=sim).collect()
    )
    log('recall_ada', question=q, hits=hits)
    lin_hits = rows(
        memories.where(memories.user_id == 'lin').order_by(sim, asc=False).limit(2)
        .select(memories.content, score=sim).collect()
    )
    log('recall_lin_same_question', hits=lin_hits)

    # --- 4. retrieval_udf on the live tables ---
    get_source = pxt.retrieval_udf(
        sources, name='get_source', description='Look up a source by id', parameters=['source_id'], limit=1
    )
    memories_of_kind = pxt.retrieval_udf(
        memories, name='memories_of_kind', description="List a user's memories of one kind",
        parameters=['user_id', 'kind'], limit=5,
    )
    enriched = rows(
        memories.where(memories.source_id != None)  # noqa: E711
        .select(memories.content, src=get_source(source_id=memories.source_id))
        .order_by(memories.created_at).collect()
    )
    log('retrieval_enrich', n=len(enriched), sample=[{'content': e['content'][:50], 'src_title': (e['src'] or [{}])[0].get('title')} for e in enriched])
    prefs = rows(sources.limit(1).select(prefs=memories_of_kind(user_id='ada', kind='preference')).collect())
    log('retrieval_by_kind', n=len(prefs[0]['prefs']) if prefs else None, keys=sorted(prefs[0]['prefs'][0].keys()) if prefs and prefs[0]['prefs'] else None)

    # --- 5. agent turns (tiny LLM pass) ---
    if os.environ.get('OPENAI_API_KEY'):
        turns.delete()
        t0 = time.time()
        st = turns.insert(
            [
                {'user_id': 'ada', 'message': 'What should my chapter say about how agents rank memories?'},
                {'user_id': 'ada', 'message': 'What year is source S-02 from, and is it background reading for me?'},
                {'user_id': 'lin', 'message': 'Remember that the reranker budget is 25 ms per query.'},
            ],
            on_error='ignore',
        )
        res = rows(turns.select(turns.user_id, turns.message, turns.memory_text, turns.tool_output, turns.answer, turns.remember).collect())
        errs = rows(turns.where(turns.response.errortype != None).select(err=turns.response.errortype, msg=turns.response.errormsg).collect())  # noqa: E711
        log('agent_turns', seconds=round(time.time() - t0, 2), status=str(st), rows=res, errors=errs)

        # write-back: proposed memories become durable rows (embedded on insert)
        new = [
            {'user_id': r['user_id'], 'kind': r['remember']['kind'], 'content': r['remember']['content'],
             'importance': r['remember']['importance'], 'created_at': dt.datetime.now(), 'source_id': None}
            for r in res if r['remember']
        ]
        if new:
            memories.insert(new)
        sim2 = memories.content.similarity(string='latency budget for reranking', idx='memory_embed')
        back = rows(memories.where(memories.user_id == 'lin').order_by(sim2, asc=False).limit(1).select(memories.content, score=sim2).collect())
        log('write_back', inserted=len(new), recall_after=back)

        # --- 6. retrieval_udf as an LLM tool in an ad-hoc query ---
        rt_tools = pxt.tools(get_source, reading_tier)  # a UDF alongside: see the note on TOOLS in app.py
        msgs = [{'role': 'user', 'content': 'Use get_source to find the title of source S-03.'}]
        t0 = time.time()
        out = rows(
            sources.limit(1).select(
                tool=openai.invoke_tools(
                    rt_tools,
                    openai.chat_completions(msgs, model=CHAT_MODEL, model_kwargs={'max_tokens': 60}, tools=rt_tools),
                )
            ).collect()
        )
        log('retrieval_udf_tool', seconds=round(time.time() - t0, 2), out=out)
    else:
        log('agent_turns', skipped='OPENAI_API_KEY not set')

    # --- 7. consolidation with UDAs ---
    prof = rows(
        memories.group_by(memories.user_id).select(
            memories.user_id,
            salience=salience(memories.importance, memories.created_at, half_life_days=30),
            digest=memory_digest(memories.kind, memories.content, memories.importance, memories.created_at),
        ).order_by(memories.user_id).collect()
    )
    log('uda_profile', rows=prof)
    by_kind = rows(
        memories.group_by(memories.user_id, memories.kind).select(
            memories.user_id, memories.kind, salience=salience(memories.importance, memories.created_at)
        ).order_by(memories.user_id, memories.kind).collect()
    )
    log('uda_salience_by_kind', rows=by_kind)
    memories.where(memories.kind == 'summary').delete()
    texts = rows(
        memories.group_by(memories.user_id).select(
            memories.user_id,
            text=digest_text(memory_digest(memories.kind, memories.content, memories.importance, memories.created_at)),
        ).collect()
    )
    summaries = [
        {'user_id': t['user_id'], 'kind': 'summary', 'content': t['text'], 'importance': 5,
         'created_at': dt.datetime.now(), 'source_id': None}
        for t in texts
    ]
    memories.insert(summaries)
    log('consolidated', summaries=[s['content'][:120] for s in summaries], total=memories.count())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
