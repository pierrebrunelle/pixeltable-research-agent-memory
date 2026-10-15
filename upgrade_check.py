#!/usr/bin/env python3
"""Check that the agent's memory survives a Pixeltable upgrade.

    python upgrade_check.py snapshot     # before: pip install -U 'pixeltable[serve]'
    python upgrade_check.py verify       # after the upgrade: same rows, same recall, writes still embed

The snapshot holds row counts, the pixeltable version, and the top recall hit for a few fixed questions.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pixeltable as pxt

ROOT = Path(__file__).resolve().parent
CATALOG = os.environ.get('MEMORY_CATALOG', 'research')
SNAP = ROOT / 'logs' / 'upgrade_snapshot.json'
QUESTIONS = [('ada', 'how do agents decide which memories matter?'), ('lin', 'latency budget for reranking')]


def state() -> dict:
    mem = pxt.get_table(f'{CATALOG}/memories')
    out = {
        'pixeltable': pxt.__version__,
        'counts': {name: pxt.get_table(f'{CATALOG}/{name}').count() for name in ('sources', 'memories', 'turns')},
        'versions': {name: pxt.get_table(f'{CATALOG}/{name}').get_metadata()['version'] for name in ('sources', 'memories', 'turns')},
        'recall': {},
    }
    for user, q in QUESTIONS:
        sim = mem.content.similarity(string=q, idx='memory_embed')
        top = mem.where(mem.user_id == user).order_by(sim, asc=False).limit(1).select(mem.content, score=sim).collect()
        out['recall'][f'{user}: {q}'] = {'content': top[0]['content'], 'score': round(float(top[0]['score']), 4)}
    return out


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    now = state()
    if cmd == 'snapshot':
        SNAP.parent.mkdir(parents=True, exist_ok=True)
        SNAP.write_text(json.dumps(now, indent=1))
        print(json.dumps(now, indent=1))
        return 0
    before = json.loads(SNAP.read_text())
    problems = []
    if now['counts'] != before['counts']:
        problems.append(f'row counts changed: {before["counts"]} -> {now["counts"]}')
    for k, v in before['recall'].items():
        if now['recall'][k]['content'] != v['content']:
            problems.append(f'recall changed for {k!r}')
    # a write after the upgrade must still be embedded and recallable
    mem = pxt.get_table(f'{CATALOG}/memories')
    mem.insert([{'user_id': 'lin', 'kind': 'fact', 'content': f'Upgrade check passed on pixeltable {pxt.__version__}',
                 'importance': 1, 'created_at': dt.datetime.now(), 'source_id': None}])
    sim = mem.content.similarity(string='did the upgrade check pass', idx='memory_embed')
    hit = mem.where(mem.user_id == 'lin').order_by(sim, asc=False).limit(1).select(mem.content).collect()[0]['content']
    if 'Upgrade check passed' not in hit:
        problems.append('post-upgrade write not recallable')
    mem.where(mem.content == f'Upgrade check passed on pixeltable {pxt.__version__}').delete()
    print(json.dumps({'from': before['pixeltable'], 'to': now['pixeltable'], 'counts': now['counts'],
                      'versions_before': before['versions'], 'versions_after': now['versions'], 'problems': problems}, indent=1))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
