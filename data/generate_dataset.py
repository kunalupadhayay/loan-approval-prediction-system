

import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

genders = np.random.choice(["Male", "Female"], size=N, p=[0.58, 0.42])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.55, 0.2, 0.15, 0.1])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.72, 0.28])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.18, 0.82])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.4, 0.35, 0.25])

# Income: lognormal for realism (in ₹/month — matches the UI's "Monthly income (₹)" field)
applicant_income = np.round(np.random.lognormal(mean=10.7, sigma=0.5, size=N)).astype(int)
applicant_income = np.clip(applicant_income, 12000, 300000)

coapplicant_income = np.where(
    married == "Yes",
    np.round(np.random.lognormal(mean=10.0, sigma=0.7, size=N)).astype(int),
    0,
)
coapplicant_income = np.clip(coapplicant_income, 0, 200000)
coapplicant_income = np.where(np.random.rand(N) < 0.25, 0, coapplicant_income)

loan_amount = np.round(np.random.lognormal(mean=3.0, sigma=0.6, size=N), 1)
loan_amount = np.clip(loan_amount, 1, 150)  # in ₹ Lakhs — matches the UI's "Loan amount (₹ Lakhs)" field

loan_term = np.random.choice([120, 180, 240, 300, 360], size=N, p=[0.05, 0.1, 0.15, 0.2, 0.5])

credit_history = np.random.choice([1, 0], size=N, p=[0.78, 0.22])  # 1 = good history

age = np.clip(np.round(np.random.normal(38, 11, N)).astype(int), 21, 70)

credit_score = np.clip(
    np.round(
        np.random.normal(650, 90, N) + np.where(credit_history == 1, 60, -60)
    ).astype(int),
    300,
    850,
)

existing_loans = np.random.choice([0, 1, 2, 3], size=N, p=[0.5, 0.28, 0.15, 0.07])

total_income = applicant_income + coapplicant_income
# Debt to income proxy: monthly EMI approx = loan_amount(in ₹ Lakhs)*100000 / term
emi = (loan_amount * 100000) / loan_term
dti = emi / (total_income + 1)

# --- Rule-driven-but-noisy approval score ---
score = (
    -0.55
    + 2.6 * credit_history
    + 0.0022 * (credit_score - 650)
    + 0.000012 * (total_income - 45000)
    - 3.0 * dti
    - 0.35 * existing_loans
    + 0.5 * (education == "Graduate").astype(int)
    - 0.25 * (self_employed == "Yes").astype(int)
    + 0.15 * (property_area == "Semiurban").astype(int)
    - 0.1 * (property_area == "Rural").astype(int)
    + np.random.normal(0, 0.9, N)  # noise
)

prob_approve = 1 / (1 + np.exp(-score))
loan_status = np.where(prob_approve > 0.5, "Y", "N")

df = pd.DataFrame(
    {
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Credit_Score": credit_score,
        "Age": age,
        "Existing_Loans": existing_loans,
        "Property_Area": property_area,
        "Debt_to_Income": np.round(dti, 4),
        "Loan_Status": loan_status,
    }
)

# introduce a small amount of realistic missingness
for col in ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    mask = np.random.rand(N) < 0.02
    df.loc[mask, col] = np.nan

out_path = "/home/claude/loan_approval_system/data/loan_dataset.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Loan_Status"].value_counts(normalize=True))
