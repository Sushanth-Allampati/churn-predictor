"""Creates a minimal test dataset for CI/CD pipelines."""
import os
import pandas as pd
import numpy as np

def create_minimal_dataset(path: str, n_rows: int = 200):
    """
    Create a minimal synthetic dataset that matches the
    Telco Churn CSV schema for use in CI tests.
    """
    np.random.seed(42)
    n = n_rows

    data = {
        'customerID'      : [f'CUST-{i:04d}' for i in range(n)],
        'gender'          : np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen'   : np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner'         : np.random.choice(['Yes', 'No'], n),
        'Dependents'      : np.random.choice(['Yes', 'No'], n),
        'tenure'          : np.random.randint(0, 72, n),
        'PhoneService'    : np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines'   : np.random.choice(
                                ['Yes', 'No', 'No phone service'], n),
        'InternetService' : np.random.choice(
                                ['DSL', 'Fiber optic', 'No'], n,
                                p=[0.34, 0.44, 0.22]),
        'OnlineSecurity'  : np.random.choice(
                                ['Yes', 'No', 'No internet service'], n),
        'OnlineBackup'    : np.random.choice(
                                ['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(
                                ['Yes', 'No', 'No internet service'], n),
        'TechSupport'     : np.random.choice(
                                ['Yes', 'No', 'No internet service'], n),
        'StreamingTV'     : np.random.choice(
                                ['Yes', 'No', 'No internet service'], n),
        'StreamingMovies' : np.random.choice(
                                ['Yes', 'No', 'No internet service'], n),
        'Contract'        : np.random.choice(
                                ['Month-to-month', 'One year', 'Two year'],
                                n, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod'   : np.random.choice([
                                'Electronic check', 'Mailed check',
                                'Bank transfer (automatic)',
                                'Credit card (automatic)'], n),
        'MonthlyCharges'  : np.round(
                                np.random.uniform(18.0, 118.0, n), 2),
        'TotalCharges'    : '',  # will be filled below
        'Churn'           : np.random.choice(
                                ['Yes', 'No'], n, p=[0.265, 0.735]),
    }

    df = pd.DataFrame(data)

    # TotalCharges = tenure * MonthlyCharges with some noise
    # 11 rows get blank TotalCharges (new customers)
    total = df['tenure'] * df['MonthlyCharges']
    total = total + np.random.normal(0, 10, n)
    total = total.clip(lower=0).round(2)
    df['TotalCharges'] = total.astype(str)
    # Set 5 rows to blank (simulate the real dataset's quirk)
    blank_idx = df[df['tenure'] == 0].index[:5]
    df.loc[blank_idx, 'TotalCharges'] = ' '

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Created {path} with {n} rows")
    return df


if __name__ == '__main__':
    create_minimal_dataset('data/raw/telco_churn.csv')