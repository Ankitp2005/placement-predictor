"""
Placement Predictor — FastAPI backend
Implements the two-stage guardrail system from the training notebook exactly:

  Stage 1 — Eligibility gate (CGPA < 6.0):
      The model has no training data below CGPA 6.5, because that population
      hasn't cleared the placement eligibility cutoff yet. Returning a fabricated
      probability would be dishonest, so we return a 'not_eligible' response
      with an explanation instead.

  Stage 2 — Model prediction + out-of-distribution flag:
      For eligible inputs, run predict_proba. If any numeric input falls outside
      the training range (e.g. CGPA 6.0–6.5 which is real but never seen by the
      model), attach a warning so the frontend can be honest about confidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"

# ---------------------------------------------------------------------------
# Guardrail constants — copied verbatim from the training notebook
# ---------------------------------------------------------------------------

# Below this CGPA, campus placement drives typically don't allow students to
# sit at all. This is a pre-model eligibility gate, not a low-confidence
# prediction — the model genuinely has nothing to say about this population.
ELIGIBILITY_CGPA_CUTOFF = 6.0

# Numeric ranges actually seen in training data — ALL features, not just
# the four picked first. Inputs outside these are still run through the
# model but flagged so the caller knows the estimate is extrapolated.
# Skills and Communication Skill Rating use the raw integer/float scales
# from the dataset (Skills: 6–9, Comm: 3–4.8), not a normalised 1–5 scale.
TRAINING_RANGES: dict[str, tuple[float, float]] = {
    "CGPA":                       (6.0, 9.5),
    "Major Projects":             (0,   2),
    "Workshops/Certificatios":    (0,   3),
    "Mini Projects":              (0,   3),
    "Skills":                     (1,  15),
    "Communication Skill Rating": (1.0, 5.0),
    "10th Percentage":            (50,  98),
    "12th Percentage":            (50,  97),
    "backlogs":                   (0,   7),
}

# Exact feature order the model was trained on.
# LightGBM stores them internally with underscores (spaces → underscores),
# but accepts DataFrames with space-separated names and converts automatically.
MODEL_FEATURE_ORDER = [
    "CGPA", "Major Projects", "Workshops/Certificatios", "Mini Projects",
    "Skills", "Communication Skill Rating", "Internship", "Hackathon",
    "12th Percentage", "10th Percentage", "backlogs",
]

# ---------------------------------------------------------------------------
# App & model
# ---------------------------------------------------------------------------
app = FastAPI(title="Placement Signal API", version="2.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class StudentFeatures(BaseModel):
    cgpa: float               = Field(..., ge=0.0,  le=10.0,  description="CGPA (0–10)")
    internship: int           = Field(..., ge=0,    le=1,     description="1=Yes, 0=No")
    hackathon: int            = Field(..., ge=0,    le=1,     description="1=Yes, 0=No")
    tenth_percentage: float   = Field(..., ge=0.0,  le=100.0, description="10th board %")
    twelfth_percentage: float = Field(..., ge=0.0,  le=100.0, description="12th board %")
    backlogs: int             = Field(..., ge=0,    le=20,    description="Number of backlogs")
    # Training data ranges: workshops 0–3, mini 0–3, major 0–2
    workshops_certifications: int = Field(..., ge=0, le=10,  description="Workshops / certifications (training range 0–3)")
    # Skills: raw integer scale 6–9 in training data (NOT a 1–5 normalised scale)
    skills: int               = Field(..., ge=0,    le=15,    description="Skills count — range 0–15")
    # Communication Skill Rating: float 3.0–4.8 in training data
    communication_skill_rating: float = Field(..., ge=1.0, le=5.0, description="Comm skill rating — training range 3.0–4.8")
    mini_projects: int        = Field(..., ge=0,    le=10,    description="Mini projects (training range 0–3)")
    major_projects: int       = Field(..., ge=0,    le=5,     description="Major projects (training range 0–2)")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Core guardrail predict function — mirrors notebook's predict_with_guardrail()
# ---------------------------------------------------------------------------
def predict_with_guardrail(model, row: dict) -> dict:
    cgpa = row.get("CGPA")

    # ── Stage 1: Eligibility gate ──────────────────────────────────────────
    # Below the cutoff, the model's output isn't a low-confidence answer —
    # it's an answer to a question that doesn't apply yet. The model was
    # never trained on this population, so we don't guess.
    if cgpa is not None and cgpa < ELIGIBILITY_CGPA_CUTOFF:
        return {
            "stage": "not_eligible",
            "probability": None,
            "probability_pct": None,
            "prediction": None,
            "in_distribution": False,
            "warnings": [],
            "message": (
                f"Most placement drives set a minimum CGPA around "
                f"{ELIGIBILITY_CGPA_CUTOFF} to be eligible to sit for interviews. "
                f"A placement-odds estimate isn't the useful number here yet — "
                f"raising CGPA past your institution's eligibility cutoff is the "
                f"first step, and clearing backlogs matters far more than any "
                f"other factor once you're eligible."
            ),
        }

    # ── Stage 2: Out-of-distribution warnings ─────────────────────────────
    # Flag inputs that fall outside what the model actually learned from.
    # These still get a prediction — just with an honesty caveat.
    warnings: list[str] = []
    for col, (lo, hi) in TRAINING_RANGES.items():
        val = row.get(col)
        if val is not None and (val < lo or val > hi):
            warnings.append(
                f"{col} = {val} is outside the training range ({lo}–{hi}). "
                f"Treat this prediction as a rough estimate, not a calibrated one."
            )

    # Build the input DataFrame in exactly the order the model was trained on
    X = pd.DataFrame([row])[MODEL_FEATURE_ORDER]
    proba = float(model.predict_proba(X)[0, 1])
    prediction = "Placed" if proba >= 0.5 else "Not Placed"

    return {
        "stage": "predicted",
        "probability": round(proba, 4),
        "probability_pct": round(proba * 100, 1),
        "prediction": prediction,
        "in_distribution": len(warnings) == 0,
        "warnings": warnings,
        "message": None,
    }


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(body: StudentFeatures) -> JSONResponse:
    model = get_model()

    # Map request body → feature dict using the column names the model expects
    row = {
        "CGPA":                     body.cgpa,
        "Major Projects":           body.major_projects,
        "Workshops/Certificatios":  body.workshops_certifications,
        "Mini Projects":            body.mini_projects,
        "Skills":                   body.skills,
        "Communication Skill Rating": body.communication_skill_rating,
        "Internship":               body.internship,
        "Hackathon":                body.hackathon,
        "12th Percentage":          body.twelfth_percentage,
        "10th Percentage":          body.tenth_percentage,
        "backlogs":                 body.backlogs,
    }

    result = predict_with_guardrail(model, row)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static file routes
# ---------------------------------------------------------------------------
@app.get("/shap_summary.png")
async def shap_image():
    path = BASE_DIR / "shap_summary.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    return JSONResponse({"error": "shap_summary.png not found"}, status_code=404)


@app.get("/")
async def root():
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
