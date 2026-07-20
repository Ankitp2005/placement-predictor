"""
generate_dataset.py
====================
Generates a realistic 10,000-student placement dataset.

Design rules:
  - CGPA < 6.0  → NotPlaced (hard eligibility gate, not trained on)
  - CGPA >= 6.0 → placement probability computed via sigmoid over all features
  - No single feature except CGPA < 6 is deterministic
  - Backlogs reduce probability significantly but cannot alone determine outcome
  - All other features (skills, comm, internship, hackathon, projects) matter meaningfully
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


# ── Feature generation helpers ──────────────────────────────────────────────

def gen_cgpa(n):
    """CGPA between 5.5 and 9.5, normally distributed around 7.5."""
    raw = RNG.normal(loc=7.5, scale=0.85, size=n)
    return np.clip(raw, 5.5, 9.5).round(1)


def gen_skills(n):
    """Number of technical skills: integer 1–15.
    Most students know 4–10 skills; distribution centred around 7."""
    raw = RNG.normal(loc=7, scale=3, size=n)
    return np.clip(np.round(raw), 1, 15).astype(int)


def gen_comm(n):
    """Communication Skill Rating: float 1.0–5.0, normally distributed around 3.5."""
    raw = RNG.normal(loc=3.5, scale=0.8, size=n)
    return np.clip(raw, 1.0, 5.0).round(1)


def gen_backlogs(n):
    """0–7, heavily weighted toward 0–3 (most students have few backlogs)."""
    return RNG.choice([0, 1, 2, 3, 4, 5, 6, 7], size=n,
                      p=[0.25, 0.28, 0.22, 0.13, 0.08, 0.02, 0.01, 0.01])


def gen_tenth(n):
    raw = RNG.normal(loc=72, scale=9, size=n)
    return np.clip(raw, 50, 98).round(1)


def gen_twelfth(n):
    raw = RNG.normal(loc=71, scale=9, size=n)
    return np.clip(raw, 50, 97).round(1)


def gen_mini_projects(n):
    return RNG.choice([0, 1, 2, 3], size=n, p=[0.22, 0.35, 0.28, 0.15])


def gen_major_projects(n):
    return RNG.choice([0, 1, 2], size=n, p=[0.30, 0.48, 0.22])


def gen_workshops(n):
    return RNG.choice([0, 1, 2, 3], size=n, p=[0.12, 0.28, 0.35, 0.25])


def gen_internship(n):
    return RNG.choice([0, 1], size=n, p=[0.42, 0.58])


def gen_hackathon(n):
    return RNG.choice([0, 1], size=n, p=[0.27, 0.73])


# ── Probability formula ─────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_placement_prob(df):
    """
    Probabilistic placement score for eligible students (CGPA >= 6.0).

    Each coefficient reflects how much that feature can shift placement odds.
    Backlogs is the strongest negative signal but NOT a cliff — a student
    with 3 backlogs and excellent other skills can still have ~25–35% odds.
    """
    cgpa  = df['CGPA'].values
    bl    = df['backlogs'].values
    sk    = df['Skills'].values
    comm  = df['Communication Skill Rating'].values
    itrn  = df['Internship'].values          # already 0/1
    hack  = df['Hackathon'].values           # already 0/1
    mini  = df['Mini Projects'].values
    major = df['Major Projects'].values
    ws    = df['Workshops/Certificatios'].values
    g12   = df['12th Percentage'].values
    g10   = df['10th Percentage'].values

    # Normalise each feature to [0, 1] range based on realistic full range
    cgpa_n  = (cgpa  - 6.0)  / 3.5          # 6.0→0, 9.5→1
    bl_n    = bl / 7                         # 0→0, 7→1  (negative)
    sk_n    = (sk    - 1)    / 14        # 1→0, 15→1
    comm_n  = (comm  - 1.0)  / 4.0      # 1.0→0, 5.0→1
    mini_n  = mini / 3                       # 0→0, 3→1
    major_n = major / 2                      # 0→0, 2→1
    ws_n    = ws / 3                         # 0→0, 3→1
    g12_n   = (g12   - 50)   / 48          # 50→0, 98→1
    g10_n   = (g10   - 50)   / 48          # 50→0, 98→1

    score = (
          0.45 * cgpa_n       # CGPA matters a lot among eligible students
        - 0.40 * bl_n         # backlogs hurt strongly but not infinitely
        + 0.20 * sk_n         # technical skills — significant
        + 0.18 * comm_n       # communication — significant
        + 0.15 * itrn         # internship — meaningful binary boost
        + 0.10 * hack         # hackathon participation
        + 0.08 * mini_n       # mini projects
        + 0.07 * ws_n         # workshops / certifications
        + 0.06 * major_n      # major projects
        + 0.04 * g12_n        # 12th grade (weaker)
        + 0.03 * g10_n        # 10th grade (weakest)
    )

    # Scale and centre: sigmoid(score*6 - 2.8)
    # At score=0.25 → sigmoid(0) = 0.5 (average student has 50% odds)
    prob = sigmoid(score * 6 - 2.8)

    # Add real-world noise (interview luck, etc.) — ±5% Gaussian, clipped
    noise = RNG.normal(0, 0.05, size=len(prob))
    prob = np.clip(prob + noise, 0.0, 1.0)

    return prob


# ── Main generation ─────────────────────────────────────────────────────────

def generate(n_total=10_000):
    print(f"Generating {n_total} student records...")

    cgpa      = gen_cgpa(n_total)
    skills    = gen_skills(n_total)
    comm      = gen_comm(n_total)
    backlogs  = gen_backlogs(n_total)
    tenth     = gen_tenth(n_total)
    twelfth   = gen_twelfth(n_total)
    mini      = gen_mini_projects(n_total)
    major     = gen_major_projects(n_total)
    ws        = gen_workshops(n_total)
    internship = gen_internship(n_total)
    hackathon  = gen_hackathon(n_total)

    df = pd.DataFrame({
        'StudentId':                  np.arange(1, n_total + 1),
        'CGPA':                       cgpa,
        'Major Projects':             major,
        'Workshops/Certificatios':    ws,
        'Mini Projects':              mini,
        'Skills':                     skills,
        'Communication Skill Rating': comm,
        'Internship':                 internship,   # 0/1 for now
        'Hackathon':                  hackathon,    # 0/1 for now
        '12th Percentage':            twelfth,
        '10th Percentage':            tenth,
        'backlogs':                   backlogs,
    })

    # ── Stage 1: hard eligibility gate ──
    ineligible_mask = df['CGPA'] < 6.0
    n_ineligible = ineligible_mask.sum()
    print(f"  Ineligible (CGPA < 6.0): {n_ineligible} students -> all NotPlaced")

    # ── Stage 2: probabilistic placement for eligible students ──
    eligible_df = df[~ineligible_mask].copy()
    prob = compute_placement_prob(eligible_df)
    placed = (RNG.random(len(eligible_df)) < prob).astype(int)

    df['PlacementStatus'] = 'NotPlaced'
    df.loc[~ineligible_mask, 'PlacementStatus'] = np.where(placed == 1, 'Placed', 'NotPlaced')
    df['_prob'] = np.nan
    df.loc[~ineligible_mask, '_prob'] = prob.round(3)

    # ── Convert binary columns back to Yes/No strings ──
    df['Internship'] = df['Internship'].map({1: 'Yes', 0: 'No'})
    df['Hackathon']  = df['Hackathon'].map({1: 'Yes', 0: 'No'})

    return df


def print_diagnostics(df):
    print("\n=== OVERALL PLACEMENT RATE ===")
    vc = df['PlacementStatus'].value_counts()
    print(vc)
    print(vc / len(df))

    print("\n=== PLACEMENT RATE BY BACKLOG COUNT ===")
    for bl in sorted(df['backlogs'].unique()):
        sub = df[df['backlogs'] == bl]
        r = (sub['PlacementStatus'] == 'Placed').mean()
        print(f"  backlogs={bl}: {r:.3f} placed  (n={len(sub)})")

    print("\n=== PLACEMENT RATE BY SKILLS (eligible only) ===")
    elig = df[df['CGPA'] >= 6.0]
    for sk in sorted(elig['Skills'].unique()):
        sub = elig[elig['Skills'] == sk]
        r = (sub['PlacementStatus'] == 'Placed').mean()
        print(f"  skills={sk}: {r:.3f} placed  (n={len(sub)})")

    print("\n=== PLACEMENT RATE: internship vs no internship (eligible) ===")
    for itrn in ['Yes', 'No']:
        sub = elig[elig['Internship'] == itrn]
        r = (sub['PlacementStatus'] == 'Placed').mean()
        print(f"  internship={itrn}: {r:.3f} placed  (n={len(sub)})")

    print("\n=== PLACEMENT RATE: hackathon vs no hackathon (eligible) ===")
    for h in ['Yes', 'No']:
        sub = elig[elig['Hackathon'] == h]
        r = (sub['PlacementStatus'] == 'Placed').mean()
        print(f"  hackathon={h}: {r:.3f} placed  (n={len(sub)})")

    print("\n=== 0 backlogs: is it still 100% placed? ===")
    z = df[df['backlogs'] == 0]
    r = (z['PlacementStatus'] == 'Placed').mean()
    print(f"  0 backlogs placement rate: {r:.3f}  (should be well below 1.0)")

    print("\n=== FEATURE RANGES (training data will use these) ===")
    elig2 = df[df['CGPA'] >= 6.0].copy()
    cols = ['CGPA','Skills','Communication Skill Rating','backlogs',
            '10th Percentage','12th Percentage',
            'Mini Projects','Major Projects','Workshops/Certificatios']
    print(elig2[cols].agg(['min','max']).round(2))


if __name__ == '__main__':
    df = generate(10_000)
    print_diagnostics(df)

    # Drop internal probability column before saving
    out = df.drop(columns=['_prob']).reset_index(drop=True)
    out.to_csv('placement_dataset_realistic.csv')
    print(f"\nSaved {len(out)} rows -> placement_dataset_realistic.csv")
