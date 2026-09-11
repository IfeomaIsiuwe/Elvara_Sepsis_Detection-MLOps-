import os
from random import seed
import pandas as pd
import numpy as np


# Feature columns
vitals_col = [
    'heart_rate',
    'temperature',
    'oxygen_saturation',
    'respiratory_rate',
    'blood_pressure'
]

labs_col = [
    'white_cell_count',
    'crp',
    'lactate',
    'creatinine',
    'platelet_count'
]


# Prediction time
def get_prediction_time(
    outcomes_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    seed: int = 7
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    outcomes = outcomes_df.copy()

    def calc_cutoff(row):
        if row['sepsis_event']:
            return row['diagnosis_time'] - pd.Timedelta(hours=9)

        pv = vitals_df[
            vitals_df['patient_id'] == row['patient_id']
        ]

        if pv.empty:
            return pd.Timestamp.now()

        start, end = pv['timestamp'].min(), pv['timestamp'].max()
        span_hours = max(
            (end - start).total_seconds() / 3600,
            1
        )

        offset = rng.uniform(0.4, 0.9) * span_hours

        return start + pd.Timedelta(hours=offset)

    outcomes['prediction_time'] = outcomes.apply(
        calc_cutoff,
        axis=1
    )

    return outcomes


# Create vital features
def extract_vital_features(
    vitals_df: pd.DataFrame,
    patient_id: int,
    cutoff: pd.Timestamp,
    lookback_hours: float = 6.0
) -> dict:

    window = vitals_df[
        (vitals_df['patient_id'] == patient_id) &
        (vitals_df['timestamp'] <= cutoff) &
        (
            vitals_df['timestamp'] >=
            cutoff - pd.Timedelta(hours=lookback_hours)
        )
    ]

    if window.empty:
        window = vitals_df[
            (vitals_df['patient_id'] == patient_id) &
            (vitals_df['timestamp'] <= cutoff)
        ].tail(1)

    feats = {}

    for col in vitals_col:

        if window.empty or col not in window:

            feats[f'{col}_mean'] = np.nan
            feats[f'{col}_min'] = np.nan
            feats[f'{col}_max'] = np.nan
            feats[f'{col}_std'] = 0.0
            feats[f'{col}_last'] = np.nan
            feats[f'{col}_rate_per_hr'] = 0.0

        else:

            vals = window[col].dropna()

            if len(vals) == 0:

                feats[f'{col}_mean'] = vals.mean()
                feats[f'{col}_min'] = vals.min()
                feats[f'{col}_max'] = vals.max()
                feats[f'{col}_std'] = (
                    vals.std()
                    if len(vals) > 1
                    else 0.0
                )
                feats[f'{col}_last'] = (
                    vals.iloc[-1]
                    if len(vals)
                    else np.nan
                )

                if len(window) > 1:

                    hours = (
                        window['timestamp'].iloc[-1] -
                        window['timestamp'].iloc[0]
                    ).total_seconds() / 3600

                    feats[f'{col}_rate_per_hr'] = (
                        (vals.iloc[-1] - vals.iloc[0]) / hours
                        if hours > 0
                        else 0.0
                    )

            else:

                feats[f'{col}_rate_per_hr'] = 0.0

    return feats


# Create lab features
def extract_lab_features(
    labs_df: pd.DataFrame,
    patient_id: int,
    cutoff: pd.Timestamp,
    lookback_hours: float = 24.0
) -> dict:

    window = labs_df[
        (labs_df['patient_id'] == patient_id) &
        (labs_df['timestamp'] <= cutoff) &
        (
            labs_df['timestamp'] >=
            cutoff - pd.Timedelta(hours=lookback_hours)
        )
    ]

    if window.empty:
        window = labs_df[
            (labs_df['patient_id'] == patient_id) &
            (labs_df['timestamp'] <= cutoff)
        ].tail(1)

    feats = {}

    for col in labs_col:

        if window.empty or col not in window:

            feats[f'{col}_mean'] = np.nan
            feats[f'{col}_last'] = np.nan

        else:

            vals = window[col].dropna()

            if len(vals) == 0:

                feats[f'{col}_mean'] = vals.mean()
                feats[f'{col}_last'] = np.nan

            else:

                feats[f'{col}_mean'] = vals.mean()
                feats[f'{col}_last'] = vals.iloc[-1]

    return feats


# Build feature matrix
def build_feature_matrix(
    patient_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    outcomes_df: pd.DataFrame
) -> pd.DataFrame:

    outcomes_with_time = get_prediction_time(
        outcomes_df,
        vitals_df
    )

    static = patient_df[
        ['patient_id', 'age', 'gender', 'medical_conditions']
    ].copy()

    static['comorbidity_count'] = static[
        'medical_conditions'
    ].apply(
        lambda x:
            0
            if pd.isna(x) or
            str(x).strip().lower()
            in ('none', 'none reported')
            else len(str(x).split(','))
    )

    static['gender_Male'] = (
        static['gender']
        .astype(str)
        .str.lower()
        .eq('male')
        .astype(int)
    )

    static = static.drop(
        columns=['gender', 'medical_conditions']
    )

    vital_rows = []

    for pid, cutoff in zip(
        outcomes_with_time['patient_id'],
        outcomes_with_time['prediction_time']
    ):

        vital_rows.append({
            'patient_id': pid,
            **extract_vital_features(
                vitals_df,
                pid,
                cutoff
            )
        })

    vital_features_df = pd.DataFrame(vital_rows)

    labs_rows = []

    for pid, cutoff in zip(
        outcomes_with_time['patient_id'],
        outcomes_with_time['prediction_time']
    ):

        labs_rows.append({
            'patient_id': pid,
            **extract_lab_features(
                labs_df,
                pid,
                cutoff
            )
        })

    labs_features_df = pd.DataFrame(labs_rows)

    features = (
        static
        .merge(
            vital_features_df,
            on='patient_id'
        )
        .merge(
            labs_features_df,
            on='patient_id'
        )
        .merge(
            outcomes_with_time[
                ['patient_id', 'sepsis_event']
            ],
            on='patient_id'
        )
    )

    feature_cols = [
        c for c in features.columns
        if c not in (
            'patient_id',
            'sepsis_event'
        )
    ]

    features[feature_cols] = (
        features[feature_cols]
        .fillna(
            features[feature_cols].median()
        )
    )

    features['sepsis_event'] = (
        features['sepsis_event'].astype(int)
    )

    return features


# Run pipeline
if __name__ == "__main__":

    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parent.parent
    SRC_DIR = BASE_DIR / "src"

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    try:
        from data_pipeline import load_raw_data, clean_data
    except ImportError:
        from src.data_pipeline import load_raw_data, clean_data

    patients, vitals, labs, history, outcomes = clean_data(
        *load_raw_data()
    )

    feat_matrix = build_feature_matrix(
        patients,
        vitals,
        labs,
        outcomes
    )

    print(
        'Engineered feature matrix shape:',
        feat_matrix.shape
    )

    print(
        'Class distribution:\n',
        feat_matrix['sepsis_event'].value_counts(
            normalize=True
        )
    )