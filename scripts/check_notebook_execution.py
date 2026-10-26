import json
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'c:\Users\admin\Documents\GitHub\RetailNova')
NOTEBOOK = ROOT / 'notebooks' / '02_eda_retailnova_storytelling.ipynb'

ns = {}
nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))

for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if not src.strip():
        continue
    print(f'Executing cell {idx}...')
    try:
        exec(compile(src, f'<cell {idx}>', 'exec'), ns, ns)
    except Exception:
        print(f'ERROR in cell {idx}')
        traceback.print_exc()
        raise

print('NOTEBOOK_OK')
