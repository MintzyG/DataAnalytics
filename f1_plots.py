'''Shared chart theme, so every notebook produces the same look.'''

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
INK, MUTED, GRID, AXIS, SURFACE = '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
DIVERGING = LinearSegmentedColormap.from_list('blue_gray_red', [BLUE, '#f0efec', RED])


def use_style():
    '''Apply the theme to the current session. Call once per notebook.'''
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'axes.titlecolor': '#0b0b0b',
        'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'text.color': INK, 'font.size': 10, 'legend.frameon': False,
    })


def corr_heatmap(data, title, ax, limit=1.0, cbar_label='correlation', **kwargs):
    '''Annotated correlation matrix on the diverging blue-gray-red scale, centred on 0.'''
    sns.heatmap(data, cmap=DIVERGING, center=0, vmin=-limit, vmax=limit, annot=True, fmt='.2f',
                annot_kws={'fontsize': 9}, linewidths=2, linecolor=SURFACE,
                cbar_kws={'label': cbar_label, 'shrink': 0.6}, ax=ax, **kwargs)
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(length=0)
