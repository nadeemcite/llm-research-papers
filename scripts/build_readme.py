"""Generates README.md, assets/banner.svg and assets/family-tree.svg from scripts/papers.tsv, and writes
the GitHub Pages (Just the Docs) nav front matter into each paper's Markdown files.

Usage (from the repo root):  python scripts/build_readme.py
Requires matplotlib. To add a paper, append a row to scripts/papers.tsv (and, optionally, its
lineage edges to EDGES below), then re-run this script.
"""
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = os.environ.get('PREVIEW_DIR')  # optional: also write PNG previews here
GOAL = 100
REPO_NAME = 'llm-research-papers'
GH_USER = 'nadeemcite'

DOMAINS = [  # (key, section title, short lane label, color)
    ('architecture', 'Architecture & Attention', 'Architecture\n& attention', '#4ea8de'),
    ('ssm', 'State Space & Linear Models', 'State space\n& linear', '#b388eb'),
    ('efficiency', 'Inference Efficiency', 'Inference\nefficiency', '#48cae4'),
    ('quantization', 'Quantization', 'Quantiz-\nation', '#80ed99'),
    ('scaling', 'Scaling Laws & Training', 'Scaling &\ntraining', '#f2c14e'),
    ('training', 'Training Methods', 'Training\nmethods', '#ffd166'),
    ('data', 'Data & Curation', 'Data &\ncuration', '#57cc99'),
    ('alignment', 'Alignment & Preference', 'Alignment\n& preference', '#ff6b9a'),
    ('reasoning', 'Reasoning', 'Reasoning', '#9d4edd'),
    ('agents', 'Agents & Tools', 'Agents\n& tools', '#ef476f'),
    ('rag', 'RAG & Retrieval', 'RAG &\nretrieval', '#06d6a0'),
    ('moe', 'Mixture of Experts', 'Mixture of\nexperts', '#e5989b'),
    ('multimodal', 'Multimodal', 'Multi-\nmodal', '#f77f00'),
    ('diffusion', 'Diffusion LMs', 'Diffusion\nLMs', '#ffd166'),
    ('interpretability', 'Interpretability', 'Interpret-\nability', '#58a6ff'),
    ('safety', 'Safety & Alignment', 'Safety &\nalignment', '#ef476f'),
]

# "built on / directly inspired by" edges between papers (by number), used for the family tree
# Add edges as papers are built and lineage becomes clear
EDGES = [
    ('003', '006'),   # SwiGLU -> LLaMA
    ('004', '006'),   # RMSNorm -> LLaMA
    ('001', '006'),   # RoPE -> LLaMA
    ('006', '007'),   # LLaMA -> Mistral
    ('006', '043'),   # LLaMA -> TinyStories
    ('016', '018'),   # S4 -> Mamba-2
    ('017', '018'),   # xLSTM -> Mamba-2
    ('020', '021'),   # Gated DeltaNet -> Kimi Linear
    ('023', '024'),   # FlashAttention-2 -> PagedAttention
    ('025', '026'),   # Speculative Decoding -> Medusa
    ('029', '030'),   # GPTQ -> AWQ
    ('030', '031'),   # AWQ -> BitNet
    ('034', '037'),   # Scaling Laws -> Superposition Scaling
    ('040', '041'),   # Self-Instruct -> LIMA
    ('041', '045'),   # LIMA -> Constitutional AI
    ('046', '047'),   # DPO -> KTO
    ('046', '048'),   # DPO -> ORPO
    ('046', '049'),   # DPO -> SimPO
    ('046', '050'),   # DPO -> Self-Rewarding
    ('052', '053'),   # CoT -> Zero-shot CoT
    ('052', '054'),   # CoT -> Self-Consistency
    ('052', '055'),   # CoT -> Tree of Thoughts
    ('052', '056'),   # CoT -> STaR
    ('056', '057'),   # STaR -> Process Reward Models
    ('057', '058'),   # Process Reward Models -> Test-Time Compute
    ('059', '060'),   # GRPO -> DeepSeek-R1
    ('059', '062'),   # GRPO -> DAPO
    ('059', '063'),   # GRPO -> Dr. GRPO
    ('060', '061'),   # DeepSeek-R1 -> s1
    ('060', '064'),   # DeepSeek-R1 -> RL vs Base Model
    ('060', '065'),   # DeepSeek-R1 -> Coconut
    ('060', '066'),   # DeepSeek-R1 -> Absolute Zero
    ('058', '061'),   # Test-Time Compute -> s1
    ('069', '070'),   # ReAct -> Toolformer
    ('069', '071'),   # ReAct -> Reflexion
    ('071', '073'),   # Reflexion -> Voyager
    ('074', '075'),   # SWE-bench -> SWE-agent
    ('075', '076'),   # SWE-agent -> AlphaEvolve
    ('077', '079'),   # HyDE -> Self-RAG
    ('079', '080'),   # Self-RAG -> GraphRAG
    ('078', '082'),   # Lost in the Middle -> Multi-Turn Lost
    ('083', '084'),   # Mixtral -> DeepSeekMoE
    ('088', '089'),   # Induction Heads -> Toy Superposition
    ('089', '090'),   # Toy Superposition -> Monosemanticity
    ('089', '091'),   # Toy Superposition -> ROME
    ('090', '093'),   # Monosemanticity -> RepEngineering
    ('093', '098'),   # RepEngineering -> Persona Vectors
    ('092', '094'),   # GCG Attack -> Sleeper Agents
    ('094', '095'),   # Sleeper Agents -> Alignment Faking
    ('095', '096'),   # Alignment Faking -> Emergent Misalignment
    ('096', '097'),   # Emergent Misalignment -> Subliminal Learning
    ('039', '046'),   # QLoRA -> DPO
    ('009', '035'),   # MLA -> DeepSeek-V3
    ('035', '033'),   # DeepSeek-V3 -> DeepSeek Sparse Attention
    ('031', '039'),   # BitNet -> QLoRA
]

BG = '#0d1117'
FG = '#e6edf3'
MUTED = '#8b949e'


def load_papers():
    with open(ROOT / 'scripts' / 'papers.tsv') as f:
        papers = list(csv.DictReader(f, delimiter='\t'))
    for p in papers:
        p['year'] = int(p['year'])
    keys = {d[0] for d in DOMAINS}
    bad = [p['num'] for p in papers if p['domain'] not in keys]
    assert not bad, f'unknown domain for {bad}'
    return papers


def draw_network(ax, w, h, seed, n=70, color='#58a6ff'):
    rng = random.Random(seed)
    pts = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(n)]
    for i, (x1, y1) in enumerate(pts):
        for x2, y2 in pts[i + 1:]:
            d = math.hypot(x1 - x2, y1 - y2)
            if d < 150:
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.18 * (1 - d / 150), lw=1, zorder=1)
    for x, y in pts:
        ax.scatter([x], [y], s=rng.uniform(6, 30), color=color, alpha=rng.uniform(0.25, 0.6), zorder=2, lw=0)


def build_banner(papers):
    w, h = 1600, 400
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    grad = [[i / 255] for i in range(256)]
    ax.imshow(grad, extent=(0, w, 0, h), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'bg', ['#1b1f4b', '#0d1117']), aspect='auto', zorder=0)
    draw_network(ax, w, h, seed=7)
    years = [p['year'] for p in papers]
    ax.text(70, 285, f'{GOAL} Papers That Changed AI — Season 2: The LLM Era', color=FG, fontsize=38, fontweight='bold', zorder=5)
    ax.text(72, 220, 'Landmark LLM-era papers reimplemented from scratch in PyTorch  ·  every notebook run end-to-end on Kaggle',
            color='#c9d1d9', fontsize=17, zorder=5)
    x = 72
    for label, color in [('Agent-generated', '#b388eb'), ('GPU-validated', '#57cc99'), ('Human-reviewed', '#f2c14e')]:
        width = 36 + 14.6 * len(label)
        ax.add_patch(FancyBboxPatch((x, 120), width, 48, boxstyle='round,pad=0,rounding_size=24',
                                    fc=color, ec='none', alpha=0.95, zorder=5))
        ax.text(x + width / 2, 144, label, color='#0d1117', fontsize=16, fontweight='bold',
                ha='center', va='center', zorder=6)
        x += width + 18
    ax.text(72, 55, f'{len(papers)} / {GOAL} papers   ·   {min(years)} → {max(years)}   ·   {len(DOMAINS)} domains',
            color=MUTED, fontsize=15, zorder=5)
    fig.savefig(ROOT / 'assets' / 'banner.svg', facecolor=BG)
    if PREVIEW_DIR:
        fig.savefig(Path(PREVIEW_DIR) / 'banner.png', facecolor=BG)
    plt.close(fig)


def build_family_tree(papers):
    years = list(range(min(p['year'] for p in papers), max(p['year'] for p in papers) + 1))
    col_w, node_w, node_h, row_h, lane_pad, left = 150, 132, 26, 32, 14, 150
    cells = defaultdict(list)
    for p in sorted(papers, key=lambda p: p['num']):
        cells[(p['domain'], p['year'])].append(p)
    lane_rows = {d[0]: max([len(v) for (dk, _), v in cells.items() if dk == d[0]] or [1]) for d in DOMAINS}
    total_h = sum(lane_rows[d[0]] * row_h + 2 * lane_pad for d in DOMAINS) + 130
    total_w = left + len(years) * col_w + 30

    fig = plt.figure(figsize=(total_w / 100, total_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, total_w); ax.set_ylim(total_h, 0); ax.axis('off')
    fig.patch.set_facecolor(BG)

    ax.text(20, 34, 'The Family Tree', color=FG, fontsize=22, fontweight='bold')
    ax.text(20, 58, 'Every paper in this repo by year and domain. Arrows show which paper built on which.',
            color=MUTED, fontsize=12.5)
    for i, y in enumerate(years):
        cx = left + i * col_w + col_w / 2
        ax.text(cx, 92, str(y), color=FG, fontsize=13, fontweight='bold', ha='center')
        ax.plot([left + i * col_w] * 2, [102, total_h - 10], color='#30363d', lw=0.8, zorder=0)

    pos, y0 = {}, 110
    colors = {d[0]: d[3] for d in DOMAINS}
    for key, _, lane_label, color in DOMAINS:
        lane_h = lane_rows[key] * row_h + 2 * lane_pad
        ax.add_patch(plt.Rectangle((8, y0 + 3), total_w - 16, lane_h - 6, fc=color, alpha=0.06, ec='none', zorder=0))
        ax.text(20, y0 + lane_h / 2, lane_label, color=color, fontsize=11.5, fontweight='bold', va='center')
        for i, yr in enumerate(years):
            for r, p in enumerate(cells.get((key, yr), [])):
                cx = left + i * col_w + col_w / 2
                cy = y0 + lane_pad + r * row_h + row_h / 2
                pos[p['num']] = (cx, cy)
        y0 += lane_h

    for a, b in EDGES:
        if a not in pos or b not in pos:
            continue
        (x1, y1), (x2, y2) = pos[a], pos[b]
        src = next(p for p in papers if p['num'] == a)
        same_col = abs(x1 - x2) < 1
        if same_col:
            start, end = (x1 + node_w / 2, y1), (x2 + node_w / 2, y2)
            rad = 0.6 if y2 > y1 else -0.6
        else:
            start, end = (x1 + node_w / 2, y1), (x2 - node_w / 2, y2)
            rad = (0.25 if y2 > y1 else -0.25) * min(1, abs(y2 - y1) / 300)
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
                                     mutation_scale=9, color=colors[src['domain']], alpha=0.55, lw=1.1, zorder=2))

    for p in papers:
        cx, cy = pos[p['num']]
        color = colors[p['domain']]
        ax.add_patch(FancyBboxPatch((cx - node_w / 2, cy - node_h / 2), node_w, node_h,
                                    boxstyle='round,pad=0,rounding_size=8', fc='#161b22', ec=color, lw=1.4, zorder=3))
        label = p['short'] if len(p['short']) <= 18 else p['short'][:17] + '…'
        ax.text(cx, cy + 0.5, label, color=FG, fontsize=9.2, ha='center', va='center', zorder=4)

    fig.savefig(ROOT / 'assets' / 'family-tree.svg', facecolor=BG)
    if PREVIEW_DIR:
        fig.savefig(Path(PREVIEW_DIR) / 'family-tree.png', facecolor=BG)
    plt.close(fig)


def build_readme(papers):
    by_domain = defaultdict(list)
    for p in papers:
        by_domain[p['domain']].append(p)
    years = [p['year'] for p in papers]
    kaggle_badge = 'https://kaggle.com/static/images/open-in-kaggle.svg'
    L = []
    L.append('<p align="center"><img src="assets/banner.svg" alt="100 Papers That Changed AI — Season 2: The LLM Era: reimplemented from scratch, '
             'validated on Kaggle" width="100%"></p>\n')
    L.append('<p align="center">')
    L.append(f'  <img alt="papers" src="https://img.shields.io/badge/papers-{len(papers)}%20%2F%20{GOAL}-4ea8de">')
    L.append('  <a href="https://www.kaggle.com/nadymsazad"><img alt="Kaggle" src="https://img.shields.io/badge/'
             'Kaggle-run%20every%20notebook-20BEFF?logo=kaggle&logoColor=white"></a>')
    L.append('  <a href="https://www.youtube.com/@BackpropDaily"><img alt="YouTube" src="https://img.shields.io/badge/'
             'YouTube-Backprop%20Daily-FF0000?logo=youtube&logoColor=white"></a>')
    L.append(f'  <a href="https://github.com/{GH_USER}/{REPO_NAME}/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-57cc99"></a>')
    L.append('</p>\n')
    L.append(f'**{len(papers)} landmark papers from the LLM era**, from RoPE ({min(years)}) to '
             f'Artificial Hivemind ({max(years)}), each turned into a small, runnable, from-scratch PyTorch '
             'reimplementation with a plain-English explainer. Every notebook in this repo was **executed '
             'top to bottom on Kaggle** and is published with its real outputs, so you can open it, see it '
             'work, and fork it on a free GPU.\n')
    L.append('## How it\'s made: agent-generated, GPU-validated, human-reviewed\n')
    L.append('This repo is also an experiment in what AI agents can build when the bar is *"it actually runs."*\n')
    L.append('1. **Curated list:** I picked 100 landmark LLM-era papers and wrote, for each one, a spec of what its '
             'notebook should demonstrate.')
    L.append('2. **Agent-generated:** an AI agent reads the paper, writes the from-scratch implementation, '
             'and drafts the explainer docs.')
    L.append('3. **GPU-validated:** every notebook is pushed to Kaggle and run end-to-end, on a T4 GPU or on CPU '
             'for the small ones. A notebook ships only if every cell runs with zero errors, and the executed '
             'version, with its outputs, is what gets committed.')
    L.append('4. **Human-reviewed:** I review the code, results and claims before publishing, and re-run or rewrite '
             'the notebooks whose results don\'t hold up.\n')
    L.append('> **Scaled down on purpose.** These are teaching-sized reproductions: small models, synthetic or small '
             'datasets, minutes of training instead of days. They show *how* each idea works and reproduce '
             'the qualitative effect, not the paper\'s headline numbers. Each `CODE_ARCHITECTURE.md` lists the '
             'simplifications made.\n')
    L.append('## The family tree\n')
    L.append('<p align="center"><a href="assets/family-tree.svg"><img src="assets/family-tree.svg" '
             'alt="Family tree of the papers in this repo, by year and domain" width="100%"></a></p>\n')
    L.append('## What\'s in each folder\n')
    L.append('| File | What it gives you |')
    L.append('|---|---|')
    L.append('| `README.md` | A 5-minute read: summary, core idea, key method details, influence, and the problem '
             'explained like you\'re 10 |')
    L.append('| `solution.ipynb` | The from-scratch implementation, executed on Kaggle with all outputs |')
    L.append('| `CODE_ARCHITECTURE.md` | A guided walkthrough of the notebook: data flow, tensor shapes, and '
             'simplifications vs. the paper |')
    L.append('| `TALK.md` | Coverage, Q&A answered from the paper, common misconceptions, and citations |\n')
    L.append('## Papers by domain\n')
    L.append(' · '.join(f'[{title}](#{anchor(title)})' for key, title, _, _ in DOMAINS if by_domain[key]) + '\n')
    for key, title, _, _ in DOMAINS:
        ps = sorted(by_domain[key], key=lambda p: (p['year'], p['num']))
        if not ps:
            continue
        L.append(f'### {title}\n')
        L.append('| # | Year | Paper | Run it |')
        L.append('|---|---|---|---|')
        for p in ps:
            L.append(f'| {p["num"]} | {p["year"]} | [{p["title"]}]({p["folder"]}/) | '
                     f'[![Open in Kaggle]({kaggle_badge})](https://www.kaggle.com/code/nadymsazad/{p["kaggle"]}) |')
        L.append('')
    L.append('<sub>Year = first public release (usually arXiv). Numbers follow the order of the curated reading list.</sub>\n')
    L.append('## Watch: Backprop Daily\n')
    L.append('Short videos on the same 100 papers: *learn AI the way models learn, one step at a time.* '
             'They\'re made by an automated pipeline that writes the script, narrates it, generates the visuals and renders the video.\n')
    L.append('More at **[youtube.com/@BackpropDaily](https://www.youtube.com/@BackpropDaily?sub_confirmation=1)**.\n')
    L.append('## About me\n')
    L.append('I\'m **Nadeem Sajjad**, a Senior AI Engineer and forward-deployed engineer with 12+ years across startups '
             'and scale-ups.\n')
    L.append('[![Website](https://img.shields.io/badge/devnadeem.com-0d1117?logo=googlechrome&logoColor=white)](https://devnadeem.com)')
    L.append(f'[![GitHub](https://img.shields.io/badge/GitHub-{GH_USER}-181717?logo=github)](https://github.com/{GH_USER})')
    L.append('[![LinkedIn](https://img.shields.io/badge/LinkedIn-nadeem--sajjad-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nadeem-sajjad/)')
    L.append('[![X](https://img.shields.io/badge/X-@nadym__sazad-000000?logo=x)](https://x.com/nadym_sazad)')
    L.append('[![YouTube](https://img.shields.io/badge/YouTube-@BackpropDaily-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@BackpropDaily)')
    L.append('[![Kaggle](https://img.shields.io/badge/Kaggle-nadymsazad-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/nadymsazad)\n')
    L.append('If this helped you understand a paper, a ⭐ helps others find it.\n')
    L.append('---\n')
    L.append(f'<sub>Licensed under the [MIT License](https://github.com/{GH_USER}/{REPO_NAME}/blob/main/LICENSE). Generated by `scripts/build_readme.py` from '
             '`scripts/papers.tsv`; edit those, not this file.</sub>')
    (ROOT / 'README.md').write_text('\n'.join(L) + '\n')


def build_nav_front_matter(papers):
    """Write Just the Docs front matter (title / parent / nav_order) into each paper's Markdown files."""
    q = lambda v: json.dumps(v, ensure_ascii=False)
    for p in papers:
        parent = f"{p['num']} {p['short']}"
        pages = [('README.md', {'title': parent, 'nav_order': int(p['num']) + 1, 'permalink': f"/{p['folder']}/"})]
        for i, (fname, label) in enumerate([('CODE_ARCHITECTURE.md', 'Code architecture'), ('TALK.md', 'Talk & Q&A')], 1):
            pages.append((fname, {'title': f"{p['short']}: {label}", 'parent': parent, 'nav_order': i}))
        for fname, fm in pages:
            path = ROOT / p['folder'] / fname
            if not path.exists():
                continue
            text = path.read_text()
            if text.startswith('---\n'):
                text = text.split('\n---\n', 1)[1]
            block = '---\n' + ''.join(f'{k}: {q(v)}\n' for k, v in fm.items()) + '---\n'
            path.write_text(block + text)


def anchor(title):
    s = ''.join(c for c in title.lower() if c.isalnum() or c in ' -')
    return s.replace(' ', '-')


if __name__ == '__main__':
    papers = load_papers()
    (ROOT / 'assets').mkdir(exist_ok=True)
    build_banner(papers)
    build_family_tree(papers)
    build_readme(papers)
    build_nav_front_matter(papers)
    print(f'Built README.md, assets/banner.svg, assets/family-tree.svg, paper nav front matter for {len(papers)} papers')