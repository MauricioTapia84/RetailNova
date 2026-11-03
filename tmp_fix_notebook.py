import json
from pathlib import Path

p = Path(r'c:\Users\admin\Documents\GitHub\RetailNova\notebooks\02_analisis_exploratorio.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
new_src = [
    "display(train_df.groupby('canal_compra', observed=True)[['cantidad_compras_12m', 'monto_promedio_compra', 'recencia_dias']].agg(['mean', 'median']).round(2))\n",
    "display(train_df.groupby('categoria_principal', observed=True)[['cantidad_compras_12m', 'monto_promedio_compra', 'proporcion_compras_promocion']].agg(['mean', 'median']).round(2))\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "sns.boxplot(data=train_df, x='canal_compra', y='monto_promedio_compra', hue=TARGET, palette=PALETTE[:2], showfliers=False, ax=axes[0])\n",
    "sns.boxplot(data=train_df, x='categoria_principal', y='monto_promedio_compra', hue=TARGET, palette=PALETTE[:2], showfliers=False, ax=axes[1])\n",
    "for ax in axes:\n",
    "    ax.tick_params(axis='x', rotation=20)\n",
    "    style_legend(ax, title='Estado de abandono', labels=('Sin abandono', 'Con abandono'))\n",
    "axes[0].set_title('Ticket promedio por canal según riesgo de abandono', fontweight='bold', pad=12)\n",
    "axes[1].set_title('Ticket promedio por categoría según riesgo de abandono', fontweight='bold', pad=12)\n",
    "axes[0].set_xlabel('Canal de compra', fontweight='bold')\n",
    "axes[0].set_ylabel('Monto promedio por compra', fontweight='bold')\n",
    "axes[1].set_xlabel('Categoría principal', fontweight='bold')\n",
    "axes[1].set_ylabel('Monto promedio por compra', fontweight='bold')\n",
    "save_figure(fig, '04_ticket_por_segmento_y_abandono.png')\n",
]
for cell in nb['cells']:
    if cell.get('id') == '#VSC-3e336868':
        cell['source'] = new_src
        print('patched cell #VSC-3e336868')
        break
else:
    raise SystemExit('cell not found')

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('saved', p)
