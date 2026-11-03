import json
from pathlib import Path

path = Path('notebooks/02_analisis_exploratorio.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'churn' in src.lower() or 'save_figure' in src.lower() and '04_ticket' in src.lower():
        print(f'--- CELL {i} ---')
        print(src[:2500])
        print('')
