from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('c:/Users/admin/Documents/GitHub/RetailNova')
sys.path.insert(0, str(ROOT / 'src'))
from paths import DATASET_PATH, GRAPHS_DIR

PALETTE = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#17BECF', '#D62728']
sns.set_theme(style='whitegrid', context='notebook')

df = pd.read_csv(DATASET_PATH)
TARGET = 'churn_6m'
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET],
)
train_df = df.loc[train_idx].copy().reset_index(drop=True)
train_df = train_df.drop_duplicates().copy()
train_df['segmento_recencia_eda'] = pd.cut(
    train_df['recencia_dias'],
    bins=[0, 30, 90, 180, 366],
    labels=['0-30 días', '31-90 días', '91-180 días', '181+ días'],
    include_lowest=True,
)

trend_specs = [
    ('segmento_recencia_eda', '09_tendencia_recencia.png', 'Clientes por segmento de recencia'),
    ('cantidad_compras_12m', '10_tendencia_compras.png', 'Clientes por compras previas'),
    ('region', '11_tendencia_region.png', 'Clientes por región'),
]

for column, filename, title in trend_specs:
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#f5f5f5')
    ax.set_facecolor('#f5f5f5')
    order = sorted(train_df[column].dropna().unique().tolist()) if pd.api.types.is_numeric_dtype(train_df[column]) else None
    sns.countplot(
        data=train_df,
        x=column,
        hue=TARGET,
        palette={0: PALETTE[0], 1: PALETTE[1]},
        ax=ax,
        order=order,
    )
    for container in ax.containers:
        for patch in container:
            value = patch.get_height()
            if value > 0:
                ax.annotate(
                    f'{value:,.0f}\n{value / len(train_df):.1%}',
                    (patch.get_x() + patch.get_width() / 2, value),
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#1f1f1f',
                )
    ax.set_title(title, fontsize=15, fontweight='bold', pad=16)
    ax.set_xlabel(column.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de clientes', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=20, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.margins(y=0.12)
    ax.legend(
        title='Estado de abandono',
        labels=('Sin abandono', 'Con abandono'),
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        title_fontsize=9,
    )
    fig.tight_layout()
    path = ROOT / 'outputs' / 'graphs' / filename
    fig.savefig(str(path), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'generated {filename} -> {path}')

print('OK')
