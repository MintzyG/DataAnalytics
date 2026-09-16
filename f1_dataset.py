'''Race-level dataset: one row per car that started a race.

Every analysis notebook starts from `load_dataset()`. The feature definitions
live here instead of in each notebook, so a change to a rolling window applies
everywhere at once.

Conventions shared by every notebook:

- Target `finish_pct`: finishing order scaled to 0 (winner) to 1 (last starter).
  Retirements rank behind classified finishers, following F1DB's
  `position_display_order`. Scaling makes a 10-car race comparable to a 39-car one.
- Pre-race information only: every form or history feature uses races *before*
  the one being predicted (`shift()` before rolling), so nothing leaks from the
  result.
'''

import sqlite3

import numpy as np
import pandas as pd

DB_PATH = 'formula.db'

# entries that never took the start; DNF, NC and DSQ stay in, those cars started
NON_STARTERS = ['DNQ', 'DNPQ', 'DNS', 'DNP', 'EX']

# the pre-race features, with the labels used in every chart and table
PRE_RACE = {
    'grid_pct': 'Grid position',
    'quali_gap_pct': 'Qualifying gap to pole',
    'con_form5': 'Car form (last 5 races)',
    'drv_form5': 'Driver form (last 5 races)',
    'con_prev_champ': "Team's championship pos. last season",
    'drv_prev_champ': "Driver's championship pos. last season",
    'drv_circuit_hist': "Driver's past results at this circuit",
    'con_dnf_rate5': 'Car retirement rate (last 5)',
    'drv_dnf_rate10': 'Driver retirement rate (last 10)',
    'drv_starts_before': 'Driver experience (starts)',
    'drv_age': 'Driver age',
    'teammate_grid_delta': 'Grid slot vs. teammate',
}

# known only after the race: never inputs to a model (see dataset.ipynb)
POST_RACE = ['points', 'laps', 'positions_gained', 'pit_stops', 'fastest_lap']


def open_db(path=DB_PATH):
    '''Read-only connection: nothing in these notebooks can modify formula.db.'''
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def prior_rolling_mean(df, by, col, window, min_periods):
    '''Mean of `col` over each group's previous `window` rows; the current row is excluded.'''
    return df.groupby(by)[col].transform(lambda s: s.shift().rolling(window, min_periods=min_periods).mean())


def load_starts(conn):
    '''One row per car that started a race, with the target and the grid scaled to 0-1.'''
    results = pd.read_sql_query('''
    SELECT rr.race_id, r.year, r.date, r.circuit_id, r.circuit_type,
           rr.position_display_order, rr.position_number, rr.position_text,
           rr.driver_id, rr.constructor_id, rr.engine_manufacturer_id,
           rr.grid_position_number AS grid,
           rr.points, rr.laps, rr.positions_gained, rr.pit_stops, rr.fastest_lap,
           d.date_of_birth
    FROM race_result rr
    JOIN race r   ON r.id = rr.race_id
    JOIN driver d ON d.id = rr.driver_id
    ''', conn)

    results = results[~results.position_text.isin(NON_STARTERS)].copy()

    results['n_starters'] = results.groupby('race_id').race_id.transform('size')
    results['finish_order'] = results.groupby('race_id').position_display_order.rank()
    results['finish_pct'] = (results.finish_order - 1) / (results.n_starters - 1)  # 0 = winner, 1 = last
    results['grid_pct'] = (results.grid - 1) / (results.n_starters - 1)
    results['dnf'] = results.position_number.isna().astype(int)  # retired, not classified or disqualified
    return results


def add_qualifying(results, conn):
    '''Attach `quali_gap_pct`: the driver's best qualifying lap relative to pole (0.01 = 1% slower).'''
    quali = pd.read_sql_query(
        'SELECT race_id, driver_id, time_millis, q1_millis, q2_millis, q3_millis FROM qualifying_result', conn)

    # single-session formats fill time_millis; knockout formats (2006+) fill q1/q2/q3 instead
    quali['best_millis'] = quali.time_millis.fillna(quali[['q1_millis', 'q2_millis', 'q3_millis']].min(axis=1))
    quali['quali_gap_pct'] = quali.best_millis / quali.groupby('race_id').best_millis.transform('min') - 1
    quali.loc[quali.quali_gap_pct > 0.2, 'quali_gap_pct'] = np.nan  # >20% off pole: no representative lap
    quali = quali.sort_values('quali_gap_pct').drop_duplicates(['race_id', 'driver_id'])

    return results.merge(quali[['race_id', 'driver_id', 'quali_gap_pct']], on=['race_id', 'driver_id'], how='left')


def add_form_features(results, conn):
    '''Attach the driver, car and history features. Rolling windows carry over between seasons.'''
    results = results.sort_values(['date', 'race_id']).reset_index(drop=True)

    # driver
    results['drv_form5'] = prior_rolling_mean(results, 'driver_id', 'finish_pct', 5, 2)
    results['drv_dnf_rate10'] = prior_rolling_mean(results, 'driver_id', 'dnf', 10, 3)
    results['drv_starts_before'] = results.groupby('driver_id').cumcount()
    results['drv_age'] = (pd.to_datetime(results.date) - pd.to_datetime(results.date_of_birth)).dt.days / 365.25
    results['drv_circuit_hist'] = (results.groupby(['driver_id', 'circuit_id']).finish_pct
                                          .transform(lambda s: s.shift().expanding().mean()))

    # car: one row per constructor per race weekend, averaged over its cars
    car = (results.groupby(['constructor_id', 'race_id', 'date'], as_index=False)
                  .agg(car_finish=('finish_pct', 'mean'), car_dnf=('dnf', 'mean'))
                  .sort_values('date'))
    car['con_form5'] = prior_rolling_mean(car, 'constructor_id', 'car_finish', 5, 2)
    car['con_dnf_rate5'] = prior_rolling_mean(car, 'constructor_id', 'car_dnf', 5, 2)
    results = results.merge(car[['constructor_id', 'race_id', 'con_form5', 'con_dnf_rate5']],
                            on=['constructor_id', 'race_id'], how='left')

    # teammate comparison within the same race
    results['teammate_grid_delta'] = results.grid - results.groupby(['race_id', 'constructor_id']).grid.transform('mean')

    # previous season's championship (a constructor can appear twice in a season with two engine suppliers)
    prev_con = pd.read_sql_query('''
        SELECT year + 1 AS year, constructor_id, MIN(position_number) AS con_prev_champ
        FROM season_constructor_standing GROUP BY year, constructor_id''', conn)
    prev_drv = pd.read_sql_query('''
        SELECT year + 1 AS year, driver_id, MIN(position_number) AS drv_prev_champ
        FROM season_driver_standing GROUP BY year, driver_id''', conn)
    results = (results.merge(prev_con, on=['year', 'constructor_id'], how='left')
                      .merge(prev_drv, on=['year', 'driver_id'], how='left'))

    results['era'] = pd.cut(results.year, bins=[1949, 1979, 2005, 2013, 2100],
                            labels=['1950–1979', '1980–2005', '2006–2013', '2014+'])
    return results


def load_dataset(path=DB_PATH):
    '''The full table every notebook analyses: starts, qualifying and pre-race features.'''
    with open_db(path) as conn:
        results = load_starts(conn)
        results = add_qualifying(results, conn)
        return add_form_features(results, conn)
