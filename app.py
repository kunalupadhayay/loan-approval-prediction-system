

import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")

app = Flask(__name__)

# --- Load model + metadata once at startup ---
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_COLUMNS = [
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Credit_Score",
    "Age",
    "Existing_Loans",
    "Debt_to_Income",
    "Property_Area",
]

REQUIRED_FIELDS = [
    "dependents",
    "education",
    "self_employed",
    "applicant_income",
    "loan_amount",
    "credit_history",
    "property_area",
]

# Fields that are genuinely optional and safe to default because the
# default reflects a neutral / typical value, not a value that flatters
# the applicant.
OPTIONAL_DEFAULTS = {
    "coapplicant_income": 0,
    "loan_term": 30,
    "credit_score": 650,
    "age": None,  # age has no safe neutral default; validated separately
    "existing_loans": 0,
}

NUMERIC_RANGES = {
    "applicant_income": (0, 10_000_000),
    "coapplicant_income": (0, 10_000_000),
    "loan_amount": (10_000, 15_000_000),  # full rupees; model uses lakhs
    "loan_term": (1, 40),  # 1 to 40 years; converted to months for the model
    "credit_score": (300, 900),
    "age": (18, 100),
    "existing_loans": (0, 20),
}

VALID_CATEGORIES = {
    "dependents": {"0", "1", "2", "3+"},
    "education": {"Graduate", "Not Graduate"},
    "self_employed": {"Yes", "No"},
    "property_area": {"Urban", "Semiurban", "Rural"},
}


class ValidationError(ValueError):
    """Raised when incoming applicant data fails validation."""


def _get(form, key):
    """Read a form value, trimming whitespace and treating empty strings as missing."""
    value = form.get(key) if form is not None else None
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    return value if value != "" else None


def _get_first(form, *keys):
    """Return the first present, non-empty value from a list of UI/API aliases."""
    for key in keys:
        value = _get(form, key)
        if value is not None:
            return value
    return None


def _to_number(value, field_name):
    try:
        # Accept UI values such as "20 years" as well as plain numbers.
        if isinstance(value, str):
            value = value.lower().replace("years", "").replace("year", "").strip()
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{field_name}' must be a number, got {value!r}.")


def validate_and_clean(form):
    """Validate current UI form fields and normalize them to the model schema."""
    if form is None:
        raise ValidationError("No form data was received. Please submit the application again.")

    # Support both the current webpage field names and older/API field names.
    aliases = {
        "dependents": ("dependents",),
        "education": ("education",),
        "self_employed": ("self_employed", "employment"),
        "applicant_income": ("applicant_income", "monthly_income"),
        "coapplicant_income": ("coapplicant_income", "co_applicant_income"),
        "loan_amount": ("loan_amount",),
        "loan_term": ("loan_term", "term"),
        "credit_score": ("credit_score",),
        "age": ("age",),
        "existing_loans": ("existing_loans",),
        "credit_history": ("credit_history",),
        "property_area": ("property_area",),
    }
    raw = {name: _get_first(form, *keys) for name, keys in aliases.items()}

    required = ("dependents", "education", "self_employed", "applicant_income",
                "loan_amount", "age", "credit_history", "property_area")
    missing = [name for name in required if raw[name] is None]
    if missing:
        # Report the user-facing name when the common UI alias was expected.
        display = {
            "self_employed": "employment",
            "applicant_income": "monthly_income",
        }
        missing_text = ", ".join(display.get(name, name) for name in missing)
        raise ValidationError(f"Missing required field(s): {missing_text}.")

    cleaned = {}

    # Employment dropdown in the current UI uses labels such as "Salaried";
    # the trained model expects Self_Employed = Yes/No.
    employment = str(raw["self_employed"]).strip()
    employment_lower = employment.lower().replace("-", " ").replace("_", " ")
    if employment_lower in {"yes", "self employed", "selfemployed", "business", "self employed / business"}:
        employment = "Yes"
    elif employment_lower in {"no", "salaried", "employed", "unemployed", "not self employed", "not self-employed"}:
        employment = "No"
    if employment not in {"Yes", "No"}:
        raise ValidationError("'employment' must be Salaried/No or Self-employed/Yes.")
    cleaned["self_employed"] = employment

    # Validate and normalize categorical values.
    cleaned["dependents"] = str(raw["dependents"]).strip()
    if cleaned["dependents"] not in VALID_CATEGORIES["dependents"]:
        raise ValidationError("'dependents' must be 0, 1, 2, or 3+.")

    cleaned["education"] = str(raw["education"]).strip()
    if cleaned["education"] not in VALID_CATEGORIES["education"]:
        raise ValidationError("'education' must be Graduate or Not Graduate.")

    area = str(raw["property_area"]).strip()
    area_key = area.lower().replace("-", "").replace(" ", "")
    area_map = {"urban": "Urban", "semiurban": "Semiurban", "rural": "Rural"}
    if area_key not in area_map:
        raise ValidationError("'property_area' must be Urban, Semi-urban, or Rural.")
    cleaned["property_area"] = area_map[area_key]

    # Numeric fields: support omitted optional fields, but never invent an age.
    def numeric_field(key, required=False):
        value = raw[key]
        if value is None:
            if required:
                raise ValidationError(f"Missing required field: '{key}'.")
            default = OPTIONAL_DEFAULTS[key]
            if default is None:
                raise ValidationError(f"Missing required field: '{key}'.")
            return float(default)
        number = _to_number(value, key)
        lo, hi = NUMERIC_RANGES[key]
        if not (lo <= number <= hi):
            raise ValidationError(f"'{key}' must be between {lo} and {hi}, got {number}.")
        return number

    cleaned["applicant_income"] = numeric_field("applicant_income", required=True)
    cleaned["coapplicant_income"] = numeric_field("coapplicant_income")
    cleaned["loan_amount"] = numeric_field("loan_amount", required=True)
    cleaned["loan_term"] = numeric_field("loan_term")
    cleaned["credit_score"] = numeric_field("credit_score")
    cleaned["age"] = numeric_field("age", required=True)
    cleaned["existing_loans"] = numeric_field("existing_loans")

    # UI labels: "Clean / on time" and "Poor / missed payments".
    credit = str(raw["credit_history"]).strip().lower().replace("_", " ")
    if credit in {"1", "1.0", "yes", "good", "clean", "clean / on time", "clean/on time", "on time", "good credit history"}:
        cleaned["credit_history"] = 1.0
    elif credit in {"0", "0.0", "no", "poor", "poor / missed payments", "poor/missed payments", "missed payments", "bad"}:
        cleaned["credit_history"] = 0.0
    else:
        raise ValidationError("'credit_history' must be Clean / on time or Poor / missed payments.")

    return cleaned

def build_input_row(cleaned):
    """Turn validated data into a single-row DataFrame matching training columns."""
    return pd.DataFrame(
        [
            {
                "Dependents": cleaned["dependents"],
                "Education": cleaned["education"],
                "Self_Employed": cleaned["self_employed"],
                "ApplicantIncome": cleaned["applicant_income"],
                "CoapplicantIncome": cleaned["coapplicant_income"],
                # UI accepts full rupee amount; training/model uses lakhs.
                "LoanAmount": cleaned["loan_amount"] / 100_000,
                "Loan_Amount_Term": cleaned["loan_term"] * 12,
                "Credit_History": cleaned["credit_history"],
                "Credit_Score": cleaned["credit_score"],
                "Age": cleaned["age"],
                "Existing_Loans": cleaned["existing_loans"],
                # Same DTI feature expected by the training dataset/model.
                "Debt_to_Income": (
                    cleaned["loan_amount"]
                    / (
                        cleaned["applicant_income"]
                        + cleaned["coapplicant_income"]
                        + 1
                    )
                ),
                "Property_Area": cleaned["property_area"],
            }
        ]
    )


def top_reasons(row_df, proba_approve):
    """Show outcome-consistent, model-input-based factors.

    These are transparency heuristics, not a claim that a single Random
    Forest feature caused the prediction. The previous version could show
    positive reasons even for a rejection because it always listed the same
    signals first.
    """
    reasons = []
    r = row_df.iloc[0]
    approved = proba_approve >= 0.5

    # Credit history is one of the strongest signals in the training data.
    if r["Credit_History"] == 0:
        reasons.append(("negative", "Credit history indicates a higher approval risk."))
    elif approved:
        reasons.append(("positive", "Clean credit history supports approval."))

    # Credit score
    score = int(r["Credit_Score"])
    if score < 580:
        reasons.append(("negative", f"Low credit score ({score}) is a risk signal."))
    elif score >= 700 and approved:
        reasons.append(("positive", f"Strong credit score ({score}) supports approval."))
    elif score < 700 and not approved:
        reasons.append(("negative", f"Credit score ({score}) is below the strong-score range."))

    # Existing debt
    loans = int(r["Existing_Loans"])
    if loans >= 2:
        reasons.append(("negative", f"You currently have {loans} existing loans."))
    elif loans == 0 and approved:
        reasons.append(("positive", "No existing loans reduces the reported debt burden."))

    # Loan size relative to annual combined income. This is a simple
    # transparency indicator; it is not presented as the model's exact DTI.
    total_monthly_income = float(r["ApplicantIncome"]) + float(r["CoapplicantIncome"])
    annual_income = total_monthly_income * 12
    loan_rupees = float(r["LoanAmount"]) * 100_000
    loan_to_annual_income = loan_rupees / max(annual_income, 1.0)
    if loan_to_annual_income > 3.0:
        reasons.append(("negative", "Requested loan amount is high relative to combined annual income."))
    elif loan_to_annual_income <= 1.0 and approved:
        reasons.append(("positive", "Requested loan amount is moderate relative to combined annual income."))

    if r["Education"] == "Graduate" and approved:
        reasons.append(("positive", "Graduate education is a positive signal in the trained data."))

    # For rejection, never pad the card with unrelated positive statements.
    if not approved and not reasons:
        reasons.append(("negative", "The model estimates approval probability below 50% for this application."))

    return reasons[:5]


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    is_json = request.is_json
    form = request.get_json() if is_json else request.form

    try:
        cleaned = validate_and_clean(form)
        row = build_input_row(cleaned)

        # Keep prediction schema exactly aligned with the trained pipeline.
        row = row[FEATURE_COLUMNS]

        proba = model.predict_proba(row)[0]
        approve_proba = float(proba[1])
        prediction = "Approved" if approve_proba >= 0.5 else "Rejected"
        reasons = top_reasons(row, approve_proba)

        result = {
            "prediction": prediction,
            "approve_probability": round(approve_proba * 100, 1),
            "reject_probability": round((1 - approve_proba) * 100, 1),
            "reasons": [{"type": t, "text": txt} for t, txt in reasons],
            "input": row.iloc[0].to_dict(),
        }
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed on the server. Check the Render application logs for details."}), 500

    if is_json:
        return jsonify(result)

    return render_template("result.html", result=result)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000)
