"""Recreate Stage 1 figures from saved results, without models or notebook analyses.

Run from any directory with the project Python environment:
    python Code/export_figures.py stability
    python Code/export_figures.py gtex
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('stability', 'gtex'))
    parser.add_argument('--source', type=Path, help='Explicit completed-result JSON snapshot')
    parser.add_argument('--output', type=Path, help='Override docs/figures/<dataset>')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    destination = (args.output or repo / 'docs' / 'figures' / args.dataset).resolve()
    source = (args.source or repo / 'docs' / 'figures' / args.dataset / 'results.json').resolve()
    if not source.is_file():
        parser.error(f'Saved results not found: {source}. No models or analyses were run.')
    notebook = repo / 'Code' / f'Stage1_{args.dataset}.ipynb'
    document = json.loads(notebook.read_text(encoding='utf-8'))
    tag = f'{args.dataset}-figure-export'
    plot_cells = [cell for cell in document['cells']
                  if cell['cell_type'] == 'code' and tag in cell.get('metadata', {}).get('tags', [])]
    if not plot_cells:
        parser.error(f'No {tag} cells found in {notebook}. No research cells were executed.')

    # Only this subprocess is configured; the Jupyter kernels are not contacted.
    os.environ['MPLBACKEND'] = 'Agg'
    if args.dataset == 'stability':
        os.environ['STAGE1_FIGURE_DIR'] = str(destination)
        os.environ['STAGE1_FIGURE_SOURCE'] = str(source)
    else:
        os.environ['GTEX_FIGURE_DIR'] = str(destination)
        os.environ['GTEX_FIGURE_SOURCE'] = str(source)
    # Each notebook's export settings provide their own imports and load its snapshot.
    namespace = {'__name__': '__figure_export__'}
    for cell in plot_cells:
        exec(compile(''.join(cell['source']), f"{notebook.name}:{cell['id']}", 'exec'), namespace)

    manifest_path = destination / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['snapshot_file'] = source.name
    manifest['snapshot_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['plot_notebook'] = notebook.relative_to(repo).as_posix()
    manifest['plot_cell_ids'] = [cell['id'] for cell in plot_cells]
    manifest['plot_source_sha256'] = hashlib.sha256(json.dumps(
        [{'id': c['id'], 'source': c['source']} for c in plot_cells],
        ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    manifest['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(destination.iterdir())
                                 if p.suffix in ('.csv', '.png', '.svg')}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Recreated figures from {source.name}: {destination}')
    print('No models, probes, or other research analyses were run.')


if __name__ == '__main__':
    main()
