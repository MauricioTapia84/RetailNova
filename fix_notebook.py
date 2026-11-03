import json
from pathlib import Path

path = Path(r'c:\Users\admin\Documents\GitHub\RetailNova\notebooks\02_analisis_exploratorio.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

patches = {
    'save_figure': '''def save_figure(figure, filename):
    path = GRAPH_DIR / filename
    if path.exists():
        path.unlink()
    figure.subplots_adjust(top=0.82, bottom=0.12, left=0.10, right=0.96)
    figure.savefig(path, dpi=220, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.show()
    plt.close(figure)
    print(f'Gráfico guardado: {path.relative_to(ROOT)}')
''',
    'annotate_count_bars': '''def annotate_count_bars(axis, denominator):
    axis.margins(y=0.16)
    max_value = max((patch.get_height() for patch in axis.patches), default=0)
    if max_value > 0:
        axis.set_ylim(0, max_value * 1.18)
    for patch in axis.patches:
        value = patch.get_height()
        if value > 0:
            axis.annotate(
                f'{value:,.0f}\\n{value / denominator:.1%}',
                (patch.get_x() + patch.get_width() / 2, value),
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                xytext=(0, 6), textcoords='offset points', clip_on=True,
            )
''',
    'target_summary': '''target_summary = train_df[TARGET].value_counts().sort_index().rename(
    index={0: 'Sin abandono', 1: 'Con abandono'}
).to_frame('clientes')

target_summary['proporcion'] = target_summary.clientes / len(train_df)
display(target_summary)

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(target_summary.index, target_summary.clientes, color=[PALETTE[0], PALETTE[1]], label='Clientes en train')
ax.bar_label(
    bars,
    labels=[f'{v:,.0f}\\n{v / len(train_df):.1%}' for v in target_summary.clientes],
    padding=6,
    fontsize=9,
    fontweight='bold',
    label_type='edge',
)
ax.set_ylim(0, max(target_summary.clientes) * 1.18)
ax.set_title('Distribución del abandono en la base de entrenamiento', fontweight='bold', pad=12)
ax.set_xlabel('Estado de abandono a 6 meses', fontweight='bold')
ax.set_ylabel('Número de clientes', fontweight='bold')
style_legend(ax, title='Estado de abandono', labels=('Sin abandono', 'Con abandono'))
save_figure(fig, '01_distribucion_objetivo_abandono.png')
''',
    'categorical_profile': '''categorical_cols = [
    column for column in train_df.columns
    if pd.api.types.is_string_dtype(train_df[column].dtype)
    or isinstance(train_df[column].dtype, pd.CategoricalDtype)
]
categorical_profile = pd.DataFrame({
    'variable': categorical_cols,
    'cardinalidad': [train_df[c].nunique(dropna=True) for c in categorical_cols],
    'faltantes': [int(train_df[c].isna().sum()) for c in categorical_cols],
    'niveles_frecuentes': [train_df[c].value_counts(dropna=False).head(5).to_dict() for c in categorical_cols],
})
display(categorical_profile)
for index, col in enumerate(categorical_cols, start=1):
    counts = train_df[col].fillna('Sin dato').value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(max(7, len(counts) * 1.15), 4.8))
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[:len(counts)], label=f'Train (n={len(train_df):,})')
    ax.bar_label(bars, labels=[f'{v:,.0f}\\n{v / len(train_df):.1%}' for v in counts], padding=5, fontsize=8, fontweight='bold', label_type='edge')
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.set(title=f'Composición de {col}: distribución de clientes por segmento', xlabel=col.replace('_', ' ').title(), ylabel='Número de clientes')
    ax.tick_params(axis='x', rotation=25)
    ax.legend(frameon=False)
    save_figure(fig, f'03_{col}_composicion.png')
''',
}

patched = []
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'def save_figure' in src:
        cell['source'] = patches['save_figure'].splitlines(keepends=True)
        patched.append('save_figure')
    if 'def annotate_count_bars' in src:
        cell['source'] = patches['annotate_count_bars'].splitlines(keepends=True)
        patched.append('annotate_count_bars')
    if 'target_summary = train_df[TARGET]' in src:
        cell['source'] = patches['target_summary'].splitlines(keepends=True)
        patched.append('target_summary')
    if 'categorical_profile = pd.DataFrame({' in src:
        cell['source'] = patches['categorical_profile'].splitlines(keepends=True)
        patched.append('categorical_profile')

if not patched:
    print('No matching plotting cells found; notebook not patched.')
else:
    print('Patched cells:', ', '.join(patched))

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Notebook saved: {path}')
