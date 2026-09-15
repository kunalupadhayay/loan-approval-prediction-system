# Loan Approval Prediction System Using Machine Learning

An end-to-end machine learning web application that predicts whether a loan
application is likely to be **Approved** or **Rejected**. The system accepts
applicant, income, credit, loan, and property details through a custom Flask
frontend and returns a prediction with a model confidence score and
easy-to-understand supporting factors.

## Project Overview

The application is designed as a demonstration of how machine learning can be
integrated into a real-world loan approval workflow.

### Main features

- Loan approval prediction using a **Random Forest Classifier**
- Flask backend for model inference
- Custom HTML/CSS frontend
- Live prediction and confidence score
- Human-readable explanation of important applicant factors
- Input validation for applicant and loan details
- Loan amount entered directly in **Indian Rupees (₹)**
- Loan tenure displayed in **years** in the user interface
- Internal conversion of the loan amount to the scale expected by the trained model
- Synthetic dataset for training and demonstration
- Pickled sklearn pipeline for deployment

## Project Structure

```text
loan_approval_system/
│
├── app.py                         # Flask backend and prediction API
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── generate_dataset.py        # Generates the synthetic training dataset
│   └── loan_dataset.csv           # Training dataset
│
├── model/
│   ├── train_model.py             # Model training pipeline
│   ├── loan_model.pkl             # Trained Random Forest pipeline
│   ├── metrics.json               # Saved evaluation metrics
│   └── feature_importance.json    # Saved feature importance information
│
├── templates/
│   ├── base.html                  # Common page layout
│   ├── index.html                 # Main loan application page
│   └── result.html                # Prediction result fallback page
│
└── static/
    ├── css/
    │   └── style.css              # Frontend styling
    └── js/
        └── script.js              # Live prediction / UI interactions
```

## Machine Learning Model

### Algorithm

The project uses a **Random Forest Classifier** from scikit-learn.

The training pipeline contains:

1. Missing-value imputation
2. Numerical feature scaling
3. Categorical feature encoding using One-Hot Encoding
4. Random Forest classification

The trained preprocessing steps and classifier are stored together as a
single sklearn `Pipeline` and saved as:

```text
model/loan_model.pkl
```

The current training configuration uses:

- **Algorithm:** Random Forest Classifier
- **Number of trees:** 300
- **Maximum depth:** 10
- **Minimum samples per leaf:** 4
- **Class weight:** balanced
- **Random state:** 42

The model-training code uses an 80/20 train-test split with stratification and
calculates accuracy, precision, recall, F1-score, ROC-AUC, and a confusion
matrix.

## Input Features

The application collects information such as:

- Gender
- Age
- Marital status
- Dependents
- Education
- Employment type
- Applicant monthly income
- Co-applicant income
- Credit score
- Credit history
- Existing loans
- Property area
- Loan amount
- Loan term

### Loan Amount

The frontend accepts the **complete loan amount in Indian Rupees**.

For example:

```text
₹5,00,000
₹15,00,000
₹1,00,00,000
```

The application internally converts the entered rupee amount into the scale
expected by the trained model before making the prediction.

### Loan Term

The user interface displays the loan tenure in **years**, for example:

```text
1 year
2 years
3 years
5 years
10 years
20 years
25 years
30 years
```

The application converts the selected tenure into months when preparing the
model input.

## Prediction Flow

The overall workflow is:

```text
User enters loan details
        ↓
Flask receives the request
        ↓
Input validation
        ↓
Data preprocessing through sklearn Pipeline
        ↓
Random Forest prediction
        ↓
Approval probability / confidence
        ↓
Approved or Rejected result
        ↓
Human-readable supporting factors
```

The main prediction endpoint is:

```text
POST /predict
```

It accepts JSON or form-encoded input and returns the prediction information
used by the frontend.

## Human-Readable Explanation

Along with the model prediction, the application displays simple factors that
help the user understand the result.

Examples include:

- Clean credit history supports approval.
- Strong credit score supports approval.
- Low credit score can be a risk signal.
- A high loan amount relative to income can increase risk.
- Existing loans can increase the reported debt burden.
- Graduate education can be shown as a supporting signal.

These explanations are **rule-based supporting messages for transparency**.
They should not be interpreted as a direct explanation of an individual
Random Forest tree or as a guaranteed reason for the model's decision.

## Dataset

The project uses a **synthetic loan application dataset** generated by
`data/generate_dataset.py`.

The dataset is intended for:

- Machine learning practice
- College/project demonstrations
- Model training and evaluation
- Flask deployment demonstrations

It does not contain real customer loan applications.

The synthetic data contains structured relationships between applicant
financial information, credit-related factors, loan requirements, and the
target loan status so that the classifier can learn meaningful patterns.

## Model Evaluation

The training script evaluates the model on a held-out test set and saves the
results in:

```text
model/metrics.json
```

The saved evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Training-set size
- Test-set size

The exact metric values should be taken from the generated
`model/metrics.json` file after training rather than assumed from an earlier
run.

## Installation

Clone or copy the project and open a terminal inside the project folder:

```bash
cd loan_approval_system
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

If you want to regenerate the dataset and train the model again:

```bash
python data/generate_dataset.py
python model/train_model.py
```

This generates/updates:

```text
data/loan_dataset.csv
model/loan_model.pkl
model/metrics.json
model/feature_importance.json
```

## Run the Flask Application

Start the application with:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

The main application is available at:

```text
/
```

The prediction API is available at:

```text
/predict
```

## Example API Request

Example JSON request:

```json
{
  "gender": "Male",
  "married": "Yes",
  "dependents": "0",
  "education": "Graduate",
  "self_employed": "No",
  "applicant_income": 60000,
  "coapplicant_income": 20000,
  "loan_amount": 1500000,
  "loan_term": 20,
  "credit_history": 1,
  "credit_score": 740,
  "age": 35,
  "existing_loans": 0,
  "property_area": "Semiurban"
}
```

The frontend uses the same prediction endpoint to update the decision panel.

## Deployment

The application can be deployed using a production WSGI server such as
Gunicorn.

Install Gunicorn:

```bash
pip install gunicorn
```

Run:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

For public deployment, configure the application for production and avoid
running Flask with debug mode enabled.

## Important Notes

- This is a **machine learning demonstration project**, not a real banking or
  lending system.
- The training data is **synthetic** and should not be treated as real-world
  underwriting data.
- A model prediction is an estimate produced from learned patterns; it is not
  a guarantee of loan approval.
- Model confidence/probability should not be interpreted as a guaranteed
  financial outcome.
- For production use, the model would require real, legally usable data,
  stronger validation, fairness testing, monitoring, security, and appropriate
  financial/regulatory review.

## Technology Stack

- **Python**
- **Flask**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **HTML**
- **CSS**
- **JavaScript**
- **Pickle**

## Project Name

**Loan Approval Prediction System Using Machine Learning**

**Application Name:** LoanWise AI
