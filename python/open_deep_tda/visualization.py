"""Self-contained offline HTML/SVG diagnostics; no plotting or JS dependencies."""

import html
import json
from pathlib import Path

import numpy as np

from .evaluation import _paired, evaluate_embedding

SCATTER_SAMPLE_CAP = 1000
_COLORS = ('#2563eb', '#dc2626', '#059669', '#7c3aed', '#d97706', '#0891b2', '#be185d', '#475569')


def _svg_start(title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 350" '
            'role="img" aria-label="' + html.escape(title, quote=True) + '">'
            '<title>' + html.escape(title) + '</title>'
            '<rect width="460" height="350" fill="white"/>'
            '<text x="24" y="24" font-size="16">' + html.escape(title) + '</text>')


def _scatter(X, labels, ids, title):
    parts = [_svg_start(title)]
    points = np.zeros((len(ids), 2))
    points[:, :min(2, X.shape[1])] = X[ids, :2]
    if len(points):
        # Uniform scaling keeps aspect ratio and avoids overflow at large units.
        unit = max(float(np.max(np.abs(points))), 1.)
        points = points / unit
        low, high = points.min(axis=0), points.max(axis=0)
        span = max(float(np.max(high - low)), np.finfo(float).tiny)
        normalized = (points - (low / 2 + high / 2)) / span
        positions = normalized * 260 + np.array([230, 190])
        categories = sorted(set(str(label) for label in labels)) if labels is not None else []
        colors = {label: _COLORS[i % len(_COLORS)] for i, label in enumerate(categories)}
        for j, ((x, y), sample_id) in enumerate(zip(positions, ids)):
            label = str(labels[j]) if labels is not None else ''
            color = colors[label] if labels is not None else _COLORS[0]
            text = f'sample {sample_id}' + (f'; label {label}' if labels is not None else '')
            parts.append(f'<circle cx="{x:.3f}" cy="{350-y:.3f}" r="2.8" fill="{color}" opacity="0.75"><title>{html.escape(text)}</title></circle>')
    parts.append('<text x="24" y="335" font-size="11">First two coordinates; shared sampled IDs</text></svg>')
    return ''.join(parts)


def _diagram_svg(entry, dimension):
    P = np.asarray(entry['source_diagram'], dtype=float).reshape(-1, 2)
    Q = np.asarray(entry['target_diagram'], dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)) or np.any(P < 0) or np.any(Q < 0):
        raise ValueError('report diagrams must have finite nonnegative endpoints')
    maximum = max(float(P.max()) if len(P) else 0., float(Q.max()) if len(Q) else 0.)
    maximum = maximum if maximum > 0 else 1.
    parts = [_svg_start(f'H{dimension} persistence diagram (raw scale)')]
    parts.append('<path d="M 50 55 V 300 H 410 M 50 300 L 295 55" stroke="#94a3b8" fill="none"/>')
    for diagram, color, name in [(P, _COLORS[0], 'reference'), (Q, _COLORS[1], 'embedding')]:
        for birth, death in diagram:
            x, y = 50 + birth / maximum * 245, 300 - death / maximum * 245
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.5" fill="{color}" opacity="0.65"><title>{name}: birth={birth:.6g}, death={death:.6g}</title></circle>')
    parts.append(f'<text x="50" y="320" font-size="11">Birth → (axis maximum {maximum:.5g}); Death ↑</text>')
    parts.append('<text x="50" y="340" font-size="11" fill="#2563eb">Reference (blue)</text><text x="220" y="340" font-size="11" fill="#dc2626">Embedding (red)</text></svg>')
    return ''.join(parts)


def _mapper_svg(graph):
    nodes = graph.get('nodes', [])[:150]
    parts = [_svg_start('Mapper cover graph')]
    groups = {}
    for node in nodes:
        groups.setdefault(int(node['interval']), []).append(node)
    positions = {}
    ordered = sorted(groups)
    for i, interval in enumerate(ordered):
        x = 50 + 360 * i / max(1, len(ordered) - 1)
        for j, node in enumerate(groups[interval]):
            y = 70 + 210 * (j + 1) / (len(groups[interval]) + 1)
            positions[node['id']] = (x, y)
    for edge in graph.get('edges', []):
        if edge['source'] in positions and edge['target'] in positions:
            a, b = positions[edge['source']], positions[edge['target']]
            parts.append(f'<line x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" y2="{b[1]:.3f}" stroke="#94a3b8" stroke-width="2"/>')
    for node in nodes:
        x, y = positions[node['id']]
        radius = min(12., 3. + np.sqrt(max(0, node['size'])))
        color = _COLORS[int(node['interval']) % len(_COLORS)]
        title = html.escape(f"node {node['id']}; interval {node['interval']}; {node['size']} samples")
        parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" fill="{color}"><title>{title}</title></circle>')
    parts.append(f'<text x="24" y="320" font-size="11">{len(nodes)} of {len(graph.get("nodes", []))} nodes; horizontal: cover interval</text>')
    parts.append('<text x="24" y="338" font-size="11">Vertical placement is schematic, not a metric embedding.</text></svg>')
    return ''.join(parts)


def save_report(reference, embedding, path, labels=None, metrics=None, mapper=None, seed=0):
    """Save an offline HTML report and return its Path.

    Scatter rendering is uniformly bounded to 1000 shared IDs. q=3 (and higher)
    is projected onto the first two coordinates for display ONLY; evaluation
    uses every supplied coordinate. Supplied evaluation metrics with finite
    H0/H1 source/target diagrams are reused without recomputing PH. Otherwise
    bounded evaluation is performed and any resource failure propagates.
    Essential/zero bars are not plotted; essential counts remain in metrics.
    All user text and JSON are HTML-escaped; no script, CDN, or service is used.
    """
    reference, embedding = _paired(reference, embedding)
    n = len(reference)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (n,):
            raise ValueError('labels must have one value per sample')
    supplied_metrics = metrics
    has_diagrams = (isinstance(metrics, dict) and isinstance(metrics.get('topology'), dict)
                    and all(isinstance(metrics['topology'].get(f'h{d}'), dict)
                            and all(key in metrics['topology'][f'h{d}'] for key in ('source_diagram', 'target_diagram'))
                            for d in (0, 1)))
    if not has_diagrams:
        metrics = evaluate_embedding(reference, embedding, seed=seed)
    count = min(n, SCATTER_SAMPLE_CAP)
    ids = np.sort(np.random.default_rng(seed).choice(n, count, replace=False)) if count < n else np.arange(n)
    selected_labels = labels[ids] if labels is not None else None
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Open Deep-TDA report</title><style>',
             'body{font:15px system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#172033;background:#f8fafc}',
             '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1em}svg{width:100%;border:1px solid #cbd5e1}',
             'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:1em}h1,h2{color:#0f172a}</style></head><body>',
             '<h1>Open Deep-TDA diagnostic report</h1>',
             '<p>Offline, self-contained HTML and SVG. No network or JavaScript dependencies.</p>',
             f'<p>Scatter: {count} of {n} rows (cap {SCATTER_SAMPLE_CAP}); shared sample IDs. '
             'For q=3 or higher, only the first two coordinates are rendered; PH uses all coordinates. '
             'Each scatter has its own display scale, so use the raw metrics to detect scale distortion.</p>',
             '<div class="grid">',
             _scatter(reference, selected_labels, ids, 'Reference scatter'),
             _scatter(embedding, selected_labels, ids, 'Embedding scatter'), '</div>',
             '<h2>Persistence diagnostics</h2>',
             '<p>Finite positive H0/H1 bars on identical sampled IDs, not full-data topology. '
             'Essential H0 and zero-length bars are omitted from plots. '
             'Raw and one global scale-aligned transport cost are recorded below; '
             'agreement of diagrams alone does not guarantee fidelity.</p><div class="grid">',
             _diagram_svg(metrics['topology']['h0'], 0), _diagram_svg(metrics['topology']['h1'], 1), '</div>',
             '<h2>Metrics and reproducibility metadata</h2><details><summary>Expand full metrics, sampling and configuration</summary><pre>',
             html.escape(json.dumps(metrics, indent=2, allow_nan=False)), '</pre></details>',
             '<h2>Scatter sampling</h2><pre>',
             html.escape(json.dumps({'seed': int(seed) if seed is not None else None, 'ids': ids.tolist(),
                                     'cap': SCATTER_SAMPLE_CAP, 'method': 'uniform without replacement'})), '</pre>']
    if supplied_metrics is not None and not has_diagrams:
        parts += ['<h2>Additional supplied metrics</h2><pre>',
                  html.escape(json.dumps(supplied_metrics, indent=2, allow_nan=False)), '</pre>']
    if mapper is not None:
        parts += ['<h2>Mapper summary (not an exact Reeb graph)</h2>',
                  '<p>Cover-preimage components and their shared sample memberships; '
                  'graph cycles are not automatically point-cloud H1 classes.</p>',
                  _mapper_svg(mapper) if isinstance(mapper, dict) and 'nodes' in mapper and 'edges' in mapper else '',
                  '<details><summary>Mapper memberships and metadata</summary><pre>',
                  html.escape(json.dumps(mapper, indent=2, allow_nan=False)), '</pre></details>']
    parts.append('</body></html>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(parts), encoding='utf-8')
    return path
