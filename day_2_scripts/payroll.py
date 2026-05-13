import os
import pandas as pd
import numpy as np
import sqlite3
import kagglehub
import re
from pathlib import Path

HERE = Path(__file__).parent


def clean_monetary_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Strip everything but digits, decimal point, and minus sign
        value = re.sub(r'[^\d.-]', '', value)
        try:
            return float(value)
        except ValueError:
            return None

    return None


def clean_percentage(value):
    if pd.isna(value) or not isinstance(value, str):
        return None

    value = value.replace('%', '')

    try:
        return float(value)
    except ValueError:
        return None


def load_payroll_data(csv_path, db_path='city_payroll.db'):
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Dataset shape: {df.shape}")

    df.columns = [col.replace(' ', '_') for col in df.columns]

    monetary_columns = [
        'Hourly_or_Event_Rate',
        'Projected_Annual_Salary', 'Q1_Payments', 'Q2_Payments', 'Q3_Payments',
        'Q4_Payments', 'Payments_Over_Base_Pay', 'Total_Payments', 'Base_Pay',
        'Permanent_Bonus_Pay', 'Longevity_Bonus_Pay', 'Temporary_Bonus_Pay',
        'Lump_Sum_Pay', 'Overtime_Pay', 'Other_Pay_&_Adjustments',
        'Other_Pay_(Payroll_Explorer)', 'Average_Health_Cost', 'Average_Dental_Cost',
        'Average_Basic_Life', 'Average_Benefit_Cost'
    ]

    for col in monetary_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_monetary_value)

    if '%_Over_Base_Pay' in df.columns:
        df['%_Over_Base_Pay'] = df['%_Over_Base_Pay'].apply(clean_percentage)

    print(f"Creating/connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)

    df.to_sql('payroll', conn, if_exists='replace', index=False)

    print("Creating indices...")
    cursor = conn.cursor()
    try:
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON payroll(Year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dept ON payroll(Department_Title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_job ON payroll(Job_Class_Title)')
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not create some indices: {e}")
        conn.rollback()

    print("\nData Summary:")

    cursor.execute('SELECT Year, COUNT(*) FROM payroll GROUP BY Year')
    years_data = cursor.fetchall()
    print("\nRecords by Year:")
    for year, count in years_data:
        print(f"  {year}: {count} records")

    cursor.execute(
        'SELECT Department_Title, COUNT(*) as count FROM payroll GROUP BY Department_Title ORDER BY count DESC LIMIT 10')
    dept_data = cursor.fetchall()
    print("\nTop 10 Department Distribution:")
    for dept, count in dept_data:
        print(f"  {dept}: {count} records")

    cursor.execute('''
        SELECT
            AVG(Total_Payments) as avg_total,
            MIN(Total_Payments) as min_total,
            MAX(Total_Payments) as max_total
        FROM payroll
    ''')
    salary_stats = cursor.fetchone()
    print("\nSalary Statistics:")
    print(f"  Average Total Payment: ${salary_stats[0]:.2f}")
    print(f"  Minimum Total Payment: ${salary_stats[1]:.2f}")
    print(f"  Maximum Total Payment: ${salary_stats[2]:.2f}")

    conn.close()
    print("\nDatabase created successfully!")

    return df


if __name__ == "__main__":
    path = kagglehub.dataset_download("cityofLA/city-payroll-data")
    print("Path to dataset files:", path)

    db_path = str(HERE / "city_payroll.db")

    csv_file = os.path.join(path, 'data.csv')

    if os.path.exists(csv_file):
        df = load_payroll_data(csv_file, db_path)
    else:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    df = load_payroll_data(csv_file, db_path)
                    break

    print("\nData Preview:")
    print(df.head())

    print("\nDetailed Dataset Summary:")

    job_titles = df['Job_Class_Title'].value_counts().head(10)
    print(f"Top 10 Job Titles (out of {df['Job_Class_Title'].nunique()} unique):")
    for title, count in job_titles.items():
        print(f"  {title}: {count}")

    df['Pay_Difference'] = df['Total_Payments'] - df['Base_Pay']

    df['Pay_Difference_%'] = pd.Series(dtype='float64')

    # Only compute % where Base_Pay > 0 to avoid division-by-zero and NaN propagation
    valid_base_pay = (df['Base_Pay'] > 0) & (~df['Base_Pay'].isna())
    if valid_base_pay.any():
        df.loc[valid_base_pay, 'Pay_Difference_%'] = (
                df.loc[valid_base_pay, 'Pay_Difference'] / df.loc[valid_base_pay, 'Base_Pay'] * 100
        )

    pay_diff = df['Pay_Difference'].dropna()
    pay_diff_pct = df['Pay_Difference_%'].dropna()

    if not pay_diff.empty:
        print("\nAverage Pay Difference: ${:.2f}".format(pay_diff.mean()))
    else:
        print("\nAverage Pay Difference: No valid data")

    if not pay_diff_pct.empty:
        print("Average Pay Difference %: {:.2f}%".format(pay_diff_pct.mean()))
    else:
        print("Average Pay Difference %: No valid data")

    if 'Average_Benefit_Cost' in df.columns:
        print("\nBenefits Analysis:")

        df['Average_Benefit_Cost'] = pd.to_numeric(df['Average_Benefit_Cost'], errors='coerce')

        benefit_costs = df['Average_Benefit_Cost'].dropna()
        benefit_costs = benefit_costs[~np.isinf(benefit_costs)]

        if not benefit_costs.empty:
            print("Average Benefits Cost: ${:.2f}".format(benefit_costs.mean()))
            print("Min Benefits Cost: ${:.2f}".format(benefit_costs.min()))
            print("Max Benefits Cost: ${:.2f}".format(benefit_costs.max()))

            print("Number of records with valid benefit data: {}".format(len(benefit_costs)))
            print("Number of records with invalid or missing benefit data: {}".format(df.shape[0] - len(benefit_costs)))

            low_benefits = benefit_costs[benefit_costs < 0].count()
            if low_benefits > 0:
                print(f"Warning: {low_benefits} records have negative benefit costs")

            zero_benefits = benefit_costs[benefit_costs == 0].count()
            if zero_benefits > 0:
                print(f"Note: {zero_benefits} records have zero benefit costs")
        else:
            print("No valid benefit cost data available.")
