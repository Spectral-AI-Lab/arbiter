"""Check local Markdown links and notebook syntax without executing training."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
files = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md',
         ROOT / 'finetuning/README.md', ROOT / 'benchmark/README.md',
         *ROOT.joinpath('docs').glob('*.md')]
for path in files:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        if not (path.parent / target.split('#')[0]).exists():
            errors.append(f'{path.relative_to(ROOT)}: missing link {target}')
for path in ROOT.joinpath('finetuning').glob('*.ipynb'):
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            if cell.get('outputs'):
                errors.append(f'{path.name}: cell {index} contains saved outputs')
            try:
                ast.parse(''.join(cell['source']))
            except SyntaxError as exc:
                errors.append(f'{path.name}: cell {index}: {exc}')
if errors:
    raise SystemExit('\n'.join(errors))
print('Documentation links and notebook syntax passed.')
