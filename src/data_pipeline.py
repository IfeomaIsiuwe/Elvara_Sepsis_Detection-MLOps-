import os
import pandas as pd
import numpy as np

def load_raw_data(data_dir: str = 'data/raw'):
    patients = pd.read_csv(
        os.path.join(data_dir, 'patients.csv'),
        parse_dates=['registration_date']
    )
    vitals = pd.read_csv(
        os.path.join(data_dir, 'vital_signs.csv'),
        parse_dates=['timestamp']
    )
    history = pd.read_csv(
        os.path.join(data_dir, 'clinical_history.csv')
    )

    labs = pd.read_csv(
        os.path.join(data_dir, 'laboratory_results.csv'),
        parse_dates=['timestamp']
    )
    outcomes = pd.read_csv(
        os.path.join(data_dir, 'sepsis_outcomes.csv'),
        parse_dates=['diagnosis_time']
    )


    return patients, vitals, history, labs, outcomes


vitals_col =  ['heart_rate',
       'temperature', 'oxygen_saturation', 'respiratory_rate',
       'blood_pressure']

labs_col = ['white_cell_count', 'crp',
       'lactate', 'creatinine', 'platelet_count']       

def clean_data(
    patients: pd.DataFrame,
    vitals: pd.DataFrame,
    history: pd.DataFrame,
    labs: pd.DataFrame,
    outcomes: pd.DataFrame
):
    patients_clean = patients.copy()
    vitals_clean = vitals.copy()
    history_clean = history.copy()
    labs_clean = labs.copy()
    outcomes_clean = outcomes.copy()

    vitals_clean = vitals_clean.drop_duplicates(subset=['patient_id', 'timestamp']).reset_index(drop=True)
    labs_clean = labs_clean.drop_duplicates(subset=['patient_id', 'timestamp']).reset_index(drop=True)

    
    vitals_clean['heart_rate'] = vitals_clean['heart_rate'].clip (30, 200)
    vitals_clean['temperature'] = vitals_clean['temperature'].clip (32, 43)
    vitals_clean['oxygen_saturation'] = vitals_clean['oxygen_saturation'].clip (50, 100)
    vitals_clean['respiratory_rate'] = vitals_clean['respiratory_rate'].clip (5, 60)
    vitals_clean['blood_pressure'] = vitals_clean['blood_pressure'].clip (40, 220)

    labs_clean['white_cell_count'] = labs_clean['white_cell_count'].clip (0.1, 50.0)
    labs_clean['crp'] = labs_clean['crp'].clip (0.0, 500.0)
    labs_clean['lactate'] = labs_clean['lactate'].clip (1.0, 20.0)
    labs_clean['creatinine'] = labs_clean['creatinine'].clip (0.1, 10.0)
    labs_clean['platelet_count'] = labs_clean['platelet_count'].clip (5.0, 700.0)


    vitals_clean[vitals_col] = vitals_clean.groupby('patient_id')[vitals_col].transform(lambda s: s.ffill())
    vitals_clean[vitals_col] = (
        vitals_clean[vitals_col]
        .fillna(
            vitals_clean.groupby('patient_id')[vitals_col]
            .transform('median')
        )
    )

    labs_clean[labs_col] = labs_clean.groupby('patient_id')[labs_col].transform(lambda s: s.ffill())
    labs_clean[labs_col] = (
        labs_clean[labs_col]
        .fillna(
            labs_clean.groupby('patient_id')[labs_col]
            .transform('median')
        )
    )

    

    return patients_clean, vitals_clean, labs_clean, history_clean, outcomes_clean

if __name__ == '__main__':
    patients, vitals, history, labs, outcomes = load_raw_data()
    p_c, v_c, h_c, l_c, o_c = clean_data(patients, vitals, history, labs, outcomes)


