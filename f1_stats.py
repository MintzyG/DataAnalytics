'''Statistical helpers shared by the analysis notebooks.

Every coefficient comes from `DataFrame.corr()`. scipy is used only for the
inference around it — the confidence interval and the p-value.

Both correlations are reported side by side, because the distance between them
is itself informative:

- **Pearson r** (`corr()`) works on the raw values and measures how close they sit
  to a *straight line*. It is exactly the R² of a one-feature least squares fit.
- **Spearman ρ** (`corr(method='spearman')`) is the same calculation on ranks, so
  it measures whether the relationship is *consistently increasing*, whatever its
  shape. Skew and outliers cannot distort it.

`|ρ| - |r|` is therefore a skew / non-linearity flag: near 0 the relationship is
straight, clearly positive means the feature carries order that a straight line
is failing to use.

Every estimate carries a 95% confidence interval and a p-value. With thousands of
rows almost any correlation is "significant", so the interval matters more than
the p-value: it shows whether two numbers are actually different.
'''

import numpy as np
import pandas as pd
from scipy import stats

# DataFrame.corr() method -> the name its coefficient goes by
METHODS = {'pearson': 'r', 'spearman': 'rho'}


def _fisher_ci(r, n, alpha):
    '''Confidence interval for a correlation, built on the arctanh scale where it is symmetric.'''
    half_width = stats.norm.isf(alpha / 2) / np.sqrt(n - 3)
    return np.tanh(np.arctanh(r) - half_width), np.tanh(np.arctanh(r) + half_width)


def _p_value(r, n):
    '''Two-sided p-value for a correlation, from the t distribution on n - 2 degrees of freedom.'''
    dof = n - 2
    return 2 * stats.t.sf(np.abs(r * np.sqrt(dof / (1 - r ** 2))), dof)


def correlations(df, cols, target='finish_pct', alpha=0.05):
    '''Pearson r and Spearman rho of each column against the target, side by side.

    `corr()` uses every row where the pair is present, so `n` varies by column.
    Against `finish_pct`, positive means a higher feature value goes with a worse finish.
    '''
    cols = list(cols)
    data = df[cols + [target]]
    n = data[cols].notna().mul(data[target].notna(), axis=0).sum()  # pairs available per column

    blocks = {}
    for method, coefficient in METHODS.items():
        r = data.corr(method=method)[target].drop(target)
        ci_low, ci_high = _fisher_ci(r, n, alpha)
        blocks[method] = pd.DataFrame({coefficient: r, 'ci_low': ci_low, 'ci_high': ci_high,
                                       'p': _p_value(r, n)})

    out = pd.concat(blocks, axis=1)
    out[('n', '')] = n
    out[('|ρ|-|r|', '')] = out[('spearman', 'rho')].abs() - out[('pearson', 'r')].abs()
    return out


def formatted(table, digits=3):
    '''Round a result table for display, keeping p-values readable in scientific notation.'''
    out = table.round(digits)
    for col in table.columns:
        name = col[-1] if isinstance(col, tuple) else col
        if name == 'p' or str(name).endswith('_p'):
            out[col] = table[col].map('{:.1e}'.format)
    return out


def partial_corr(df, col, controls, target='finish_pct', method='spearman'):
    '''Correlation between `col` and the target after removing the linear effect of `controls`.

    Both columns are regressed on the controls and what is left over is correlated.
    `spearman` replaces every column by its ranks first, exactly as `corr(method='spearman')` does.
    '''
    d = df[[col, target] + controls].dropna()
    if method == 'spearman':
        d = d.rank()
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in controls])
    residual = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    left_over = pd.DataFrame({col: residual(d[col].to_numpy()), target: residual(d[target].to_numpy())})

    r = left_over.corr()[target][col]  # residuals are already ranked if they needed to be
    dof = len(d) - 2 - len(controls)
    p = 2 * stats.t.sf(abs(r * np.sqrt(dof / (1 - r ** 2))), dof)
    return pd.Series({f'partial_{METHODS[method]}': r, 'partial_p': p})


def partial_correlations(df, cols, controls, target='finish_pct'):
    '''`partial_corr` for every column and both methods, shaped to sit beside `correlations`.'''
    return pd.concat({method: pd.DataFrame([partial_corr(df, c, controls, target, method) for c in cols],
                                           index=list(cols))
                      for method in METHODS}, axis=1)


def eta_squared(df, keys, target='finish_pct'):
    '''Variance of the target explained by the group means of `keys`, raw and bias-corrected.

    IDs have no order, so no correlation applies to them. This is the categorical
    equivalent: the R² of a model that predicts each group's mean.

    A grouping with many small groups fits noise, so even a meaningless split scores
    (k - 1) / (n - 1). Subtracting it gives epsilon squared. Kruskal-Wallis tests whether
    the groups differ at all.
    '''
    d = df.dropna(subset=[target])
    codes = d.groupby(keys, observed=True).ngroup()
    y = d[target]
    k, n = codes.nunique(), len(d)
    eta2 = ((y.groupby(codes).transform('mean') - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    null = (k - 1) / (n - 1)
    _, p = stats.kruskal(*[g.to_numpy() for _, g in y.groupby(codes)])
    return pd.Series({'groups': k, 'n': n, 'eta2': eta2, 'null_eta2': null,
                      'eps2': (eta2 - null) / (1 - null), 'kruskal_p': p})


def mean_ci(values, alpha=0.05):
    '''Mean with a t confidence interval.'''
    lo, hi = stats.t.interval(1 - alpha, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
    return pd.Series({'mean_rho': values.mean(), 'ci_low': lo, 'ci_high': hi, 'races': len(values)})
