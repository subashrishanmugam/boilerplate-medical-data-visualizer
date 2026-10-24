import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Overweight column (BMI > 25 → 1, else 0)
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize: 1 → 0 (good), >1 → 1 (bad)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Categorical plot
def draw_cat_plot():
    # 5. melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. group and count
    df_cat = (df_cat.groupby(['cardio', 'variable', 'value'])
                    .size()
                    .reset_index(name='total'))

    # 7. catplot
    g = sns.catplot(
        data=df_cat, kind='bar',
        x='variable', y='total', hue='value', col='cardio'
    )

    # 8. figure
    fig = g.fig
    fig.savefig('catplot.png')
    return fig


# 10. Heat map
def draw_heat_map():
    # 11. Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Correlation matrix
    corr = df_heat.corr()

    # 13. Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0,
                vmax=0.32, vmin=-0.16, square=True, linewidths=.5,
                cbar_kws={'shrink': .5}, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig