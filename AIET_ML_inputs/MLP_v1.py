# ==========================================================
# AIET Exoplanet Pipeline v1: Physics-First ML Training
# ==========================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
BASE_DIR = r"C:\Users\LuchK\AIET_ML_inputs"
PLANETS_FILE = os.path.join(BASE_DIR, "exoplanets.csv")
STARS_FILE   = os.path.join(BASE_DIR, "stellar_hosts.csv")

OUTPUT_FILE        = os.path.join(BASE_DIR, "MLP_v1_scored.csv")
OUTPUT_SAMPLE_FILE = os.path.join(BASE_DIR, "MLP_v1_100scored.csv")
BAR_GRAPH_FILE     = os.path.join(BASE_DIR, "MLP_v1_bar_graph.png")
MODEL_SAVE_PATH    = os.path.join(BASE_DIR, "hab_net_v1.pth")
SCALER_SAVE_PATH   = os.path.join(BASE_DIR, "scaler_v1.joblib")

# -------------------------
# LOAD DATA
# -------------------------
print(">>> Loading Data...")
planets_df = pd.read_csv(PLANETS_FILE, comment="#", low_memory=False)
stars_df   = pd.read_csv(STARS_FILE, comment="#", low_memory=False)

if "default_flag" in planets_df.columns:
    planets_df = planets_df[planets_df["default_flag"] == 1]

df = pd.merge(planets_df, stars_df, on="hostname", how="left")
print(f"Data Loaded: {len(df)} exoplanets.")

# ==========================================================
# PHASE 1 — ASTROPHYSICAL IMPUTATION
# ==========================================================
print(">>> Phase 1: Cleaning Data with Astrophysics...")

bins   = [0, 3700, 5200, 6000, 7500, 50000]
labels = ["M", "K", "G", "F", "Hot"]
df["spectral_bin"] = pd.cut(df["st_teff"], bins=bins, labels=labels)

for col in ["st_mass", "st_rad", "st_lum"]:
    df[col] = df[col].fillna(
        df.groupby("spectral_bin", observed=True)[col].transform("median")
    )
    df[col] = df[col].fillna(df[col].median())

# Stefan–Boltzmann enforcement
df["st_lum"] = df["st_lum"].fillna(
    (df["st_rad"]**2) * (df["st_teff"] / 5778.0)**4
)

# Planet radius and orbital parameters
df["pl_rade"]   = df["pl_rade"].fillna(df["pl_rade"].median())
df["pl_orbper"] = df["pl_orbper"].fillna(df["pl_orbper"].median())
df["pl_orbeccen"] = df["pl_orbeccen"].fillna(0.0)

# Rocky-only mass-radius relation
mask_rocky = (df["pl_rade"] < 1.6) & (df["pl_masse"].isna())
df.loc[mask_rocky, "pl_masse"] = df.loc[mask_rocky, "pl_rade"]**2.06

# ==========================================================
# PHASE 2 — FEATURE ENGINEERING
# ==========================================================
print(">>> Phase 2: Calculating Derived Physics...")

G        = 6.67430e-11
M_earth  = 5.972e24
R_earth  = 6.371e6

df["pl_mass_kg"]  = df["pl_masse"] * M_earth
df["pl_radius_m"] = df["pl_rade"] * R_earth

# Density (g/cm^3)
df["density_calc"] = (
    df["pl_mass_kg"] / ((4/3) * np.pi * df["pl_radius_m"]**3)
) / 1000.0
df["pl_dens"] = df["pl_dens"].fillna(df["density_calc"])

# Escape velocity (m/s)
df["v_esc"] = np.sqrt(2 * G * df["pl_mass_kg"] / df["pl_radius_m"])

# Semi-major axis (AU)
df["P_yr"] = df["pl_orbper"] / 365.25
df["a_calc"] = (df["P_yr"]**2 * df["st_mass"])**(1/3)
df["pl_orbsmax"] = df["pl_orbsmax"].fillna(df["a_calc"])

# Insolation flux
df["flux_calc"] = df["st_lum"] / (df["pl_orbsmax"]**2)
df["pl_insol"] = df["pl_insol"].fillna(df["flux_calc"])

# ==========================================================
# PHASE 3 — TIDAL LOCKING (SIGMOID)
# ==========================================================
df["tidal_lock_score"] = 1 / (1 + np.exp(-(df["pl_orbper"] - 25)/5))

# ==========================================================
# PHASE 4 — PHYSICS TEACHER
# ==========================================================
print(">>> Phase 3: Generating Ground Truth Scores...")

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

df["Habitability_Physics_Percent"] = (
    0.35 * gaussian(df["pl_insol"], 1.0, 0.4) +
    0.25 * gaussian(df["pl_rade"], 1.0, 0.3) +
    0.15 * gaussian(df["pl_dens"], 5.5, 2.0) +
    0.15 * gaussian(df["v_esc"], 11186, 4000) +
    0.10 * df["tidal_lock_score"]
).clip(0,1) * 100

# ==========================================================
# PHASE 5 — ML STUDENT
# ==========================================================
print(">>> Phase 4: Training Neural Network (The Student)...")

feature_cols = [
    "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
    "pl_insol", "st_teff", "st_mass", "st_rad", "st_lum"
]

mask_finite = np.isfinite(df[feature_cols]).all(axis=1) & \
              np.isfinite(df["Habitability_Physics_Percent"])

df_train = df[mask_finite]

X = df_train[feature_cols].values.astype(np.float32)
y = df_train["Habitability_Physics_Percent"].values.astype(np.float32).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f">>> Scaler saved to: {SCALER_SAVE_PATH}")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(len(feature_cols),64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    df.loc[mask_finite, "Habitability_ML_Percent"] = (
        model(
            torch.tensor(
                scaler.transform(df.loc[mask_finite, feature_cols].values),
                dtype=torch.float32
            )
        ).numpy().flatten()
    )

# Clip ML outputs to 0–100%
df["Habitability_ML_Percent"] = df["Habitability_ML_Percent"].clip(0,100)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# ==========================================================
# PHASE 6 — EXPORT
# ==========================================================
print(">>> Phase 5: Saving Results...")

df.to_csv(OUTPUT_FILE, index=False)
df.sample(100, random_state=42).to_csv(OUTPUT_SAMPLE_FILE, index=False)

# ==========================================================
# PHASE 7 — RANDOM 5 PLANETS HABITABILITY SCORE BAR GRAPH
# ==========================================================
print(">>> Phase 6: Creating Bar Graph...")

# Get planets with valid habitability scores
valid_df = df[df["Habitability_ML_Percent"].notna()].copy()
valid_df["Habitability_ML_Percent"] = valid_df["Habitability_ML_Percent"].clip(0.0, 100.0)

if len(valid_df) > 0:
    # Randomly select 5 planets (or all if fewer than 5)
    n_select = min(5, len(valid_df))
    selected = valid_df.sample(n=n_select, random_state=42)
    
    # Sort by habitability score for better visualization
    selected = selected.sort_values("Habitability_ML_Percent", ascending=True)
    
    # Create bar graph
    plt.figure(figsize=(10, 6))
    
    # Get planet names (use hostname + pl_name if available, else just hostname)
    planet_names = []
    for idx, row in selected.iterrows():
        if pd.notna(row.get("pl_name")):
            name = f"{row['pl_name']}"
        elif pd.notna(row.get("hostname")):
            name = f"{row['hostname']}"
        else:
            name = f"Planet {idx}"
        planet_names.append(name)
    
    scores = selected["Habitability_ML_Percent"].values
    
    # Color bars based on habitability score
    colors = []
    for score in scores:
        if score >= 50:
            colors.append('#2ecc71')  # Green
        elif score >= 25:
            colors.append('#f39c12')  # Orange/Yellow
        else:
            colors.append('#e74c3c')  # Red
    
    # Create bar plot
    bars = plt.barh(range(len(planet_names)), scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}%', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Customize plot
    plt.yticks(range(len(planet_names)), planet_names)
    plt.xlabel(
        'Habitability Score (%)\n'
        'Relative, normalized habitability scores for comparative analysis (not a life-detection metric).',
        fontsize=12,
        fontweight='bold'
    )
    plt.ylabel('Planet', fontsize=12, fontweight='bold')
    plt.title('Random 5 Planets Habitability Score Comparison', fontsize=14, fontweight='bold')
    plt.xlim(0, max(scores) * 1.15 if max(scores) > 0 else 100)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(BAR_GRAPH_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar graph to: {BAR_GRAPH_FILE}")
    
    # Print selected planets info
    print("\nSelected Planets:")
    for name, score in zip(planet_names, scores):
        print(f"  {name}: {score:.2f}%")
else:
    print("No planets with valid habitability scores to plot.")

print("\n>>> Pipeline Complete.")
