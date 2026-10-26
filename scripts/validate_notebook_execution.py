import json
import traceback
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.show = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / 'notebooks' / '02_eda_retailnova_storytelling.ipynb'

ns = {}
nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))

for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    print(f'\n--- executing cell {idx} ---')
    try:
        exec(compile(src, f'<cell {idx}>', 'exec'), ns, ns)
        print('OK')
    except Exception:
        print(f'ERROR in cell {idx}')
        traceback.print_exc()
        raise

print('\nNotebook validation passed.')
