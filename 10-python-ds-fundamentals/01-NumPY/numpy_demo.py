"""
Dataset reale: dati di allenamento esportati e aggiornati da Fitbod (marzo 2025 → aprile 2026).
Ultimo aggiornamento: aprile 2026 (229 sessioni, 3941 set registrati, 86 esercizi unici)
Focus: operazioni fondamentali di NumPy su dati reali, con occhio a ML/AI.
"""

# Setup ambiente (venv + dipendenze): vedi ../SETUP.md

import csv
import timeit
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Path al dataset, ricavato a partire dalla posizione di questo file:
# .../data-science/10-python-ds-fundamentals/01-NumPY/numpy_demo.py
#  → parent×3 = .../data-science/  →  /00-datasets/workoutExport.csv
# In questo modo lo script funziona indipendentemente dalla cwd da cui lo lanci.
WORKOUT_CSV = Path(__file__).resolve().parents[2] / "00-datasets" / "workoutExport.csv"

# 0. CARICAMENTO DATI
# Prima di fare qualsiasi cosa con NumPy, carichiamo il CSV con la stdlib.
with open(WORKOUT_CSV) as f:
    raw = list(csv.DictReader(f))

print(f"Righe totali:      {len(raw)}")
print(f"Colonne:           {list(raw[0].keys())}")
print(f"Prima riga:        {raw[0]}")


# ===========================================================================
# 1. PERCHÉ NUMPY — IL PROBLEMA CON LE LISTE PYTHON
# ===========================================================================
# Vogliamo calcolare il volume totale (kg × reps) di ogni set.
# Con liste Python pura:

volumes_list = [
    float(r["Weight(kg)"]) * int(r["Reps"])
    for r in raw
    if float(r["Weight(kg)"]) > 0 and int(r["Reps"]) > 0
]
print(f"\nSet con peso e reps: {len(volumes_list)}")
print(f"Volume totale (lista): {sum(volumes_list):,.0f} kg")

# Con NumPy — stessa operazione, vettorizzata:
strength_rows = [r for r in raw if float(r["Weight(kg)"]) > 0 and int(r["Reps"]) > 0]

weights = np.array([float(r["Weight(kg)"]) for r in strength_rows])
reps = np.array([int(r["Reps"]) for r in strength_rows])

# Moltiplicazione element-wise in C, nessun loop Python
volumes = weights * reps
print(f"Volume totale (NumPy): {volumes.sum():,.0f} kg")

# Benchmark: su dataset piccolo la differenza è minima, ma su milioni di righe
# (es. log di allenamenti di una piattaforma intera) NumPy è 10x–100x più veloce.


# ===========================================================================
# 2. ATTRIBUTI FONDAMENTALI
# ===========================================================================

print(f"\nweights shape:  {weights.shape}")  # (3628,) — vettore 1D
print(f"weights dtype:  {weights.dtype}")  # float64
print(f"weights nbytes: {weights.nbytes} B")  # 3628 * 8 = 29024 byte

# In ML questi sarebbero le feature di un dataset:
# ogni riga = un set, ogni colonna = una feature (peso, reps, volume...)
# Costruiamo una feature matrix (n_samples × n_features)
X = np.column_stack([weights, reps, volumes])
print(f"\nFeature matrix X shape: {X.shape}")  # (3628, 3)
print("Colonne: [weight_kg, reps, volume]")
print(f"Prime 5 righe:\n{X[:5]}")


# ===========================================================================
# 3. DTYPE E MEMORIA
# ===========================================================================
# NumPy usa float64 di default. I framework ML (PyTorch, TensorFlow)
# usano float32 — metà della memoria, abbastanza preciso per i modelli.

X_f32 = X.astype(np.float32)
print(f"\nX float64: {X.nbytes} byte")
print(f"X float32: {X_f32.nbytes} byte  (metà)")


# ===========================================================================
# 4. STATISTICHE DESCRITTIVE — AXIS
# ===========================================================================
# axis=0 → operazione su righe (risultato per colonna)
# axis=1 → operazione su colonne (risultato per riga)

print("\n--- Statistiche per feature (axis=0) ---")
feature_names = ["weight_kg", "reps", "volume"]
for i, name in enumerate(feature_names):
    col = X[:, i]
    print(
        f"{name:12s}  mean={col.mean():8.2f}  std={col.std():7.2f}  "
        f"min={col.min():6.1f}  max={col.max():6.1f}"
    )


# ===========================================================================
# 5. INDEXING E BOOLEAN INDEXING
# ===========================================================================
# Estraiamo tutti i set di Barbell Bench Press e analizziamo la progressione.

exercise_names = np.array([r["Exercise"] for r in strength_rows])
dates_str = np.array([r["Date"][:10] for r in strength_rows])

# Boolean indexing — filtra i set di bench press
bench_mask = exercise_names == "Barbell Bench Press"
bench_weights = weights[bench_mask]
bench_dates = dates_str[bench_mask]

print(f"\nBarbell Bench Press: {bench_mask.sum()} set")
print(f"Peso min: {bench_weights.min()} kg  |  max: {bench_weights.max()} kg")
print(f"Media:    {bench_weights.mean():.1f} kg  |  std: {bench_weights.std():.1f} kg")

# Stessa cosa per Lat Pulldown
lat_mask = exercise_names == "Lat Pulldown"
lat_weights = weights[lat_mask]
print(f"\nLat Pulldown: {lat_mask.sum()} set")
print(f"Peso min: {lat_weights.min()} kg  |  max: {lat_weights.max()} kg")
print(f"Media:    {lat_weights.mean():.1f} kg  |  std: {lat_weights.std():.1f} kg")


# ===========================================================================
# 6. RESHAPE E BROADCASTING — NORMALIZZAZIONE
# ===========================================================================
# Prima di passare dati a un modello ML, si normalizzano le feature.
# Con numpy questo si fa con broadcasting — nessun loop.

# Min-Max scaling: porta ogni colonna in [0, 1]
X_min = X.min(axis=0)  # shape (3,) — min per colonna
X_max = X.max(axis=0)  # shape (3,) — max per colonna
X_norm = (X - X_min) / (X_max - X_min)  # broadcasting: (3628,3) - (3,) → ok

print("\nDopo Min-Max scaling:")
print(f"  min per colonna: {X_norm.min(axis=0)}")
print(f"  max per colonna: {X_norm.max(axis=0)}")

# Z-score standardization: media=0, std=1
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std_scaled = (X - X_mean) / X_std

print("\nDopo Z-score:")
print(f"  media per colonna: {X_std_scaled.mean(axis=0).round(6)}")
print(f"  std per colonna:   {X_std_scaled.std(axis=0).round(6)}")


# ===========================================================================
# 7. AGGREGAZIONI SETTIMANALI — VOLUME NEL TEMPO
# ===========================================================================
# Calcoliamo il volume totale per settimana — utile per vedere l'andamento
# dell'intensità degli allenamenti nel tempo.

# Raggruppiamo per settimana con Python, poi usiamo NumPy per i calcoli
weekly_volumes = defaultdict(float)
weekly_sets = defaultdict(int)

for r in strength_rows:
    d = datetime.strptime(r["Date"][:10], "%Y-%m-%d")
    week = d.strftime("%Y-W%W")
    vol = float(r["Weight(kg)"]) * int(r["Reps"])
    weekly_volumes[week] += vol
    weekly_sets[week] += 1

weeks = sorted(weekly_volumes.keys())
vol_array = np.array([weekly_volumes[w] for w in weeks])
sets_array = np.array([weekly_sets[w] for w in weeks])

print(f"\nSettimane totali: {len(weeks)}")
print(f"Volume medio settimanale: {vol_array.mean():,.0f} kg")
print(f"Volume max:  settimana {weeks[vol_array.argmax()]} — {vol_array.max():,.0f} kg")
print(f"Volume min:  settimana {weeks[vol_array.argmin()]} — {vol_array.min():,.0f} kg")
print(f"Set medi/settimana: {sets_array.mean():.1f}")

# argmax/argmin restituiscono l'indice del massimo/minimo — in ML si usa
# per trovare la classe con probabilità massima dopo un softmax.


# ===========================================================================
# 8. CONFRONTO PERIODI — PRE/POST PT
# ===========================================================================
# Dal 19 marzo 2026 hai un personal trainer. Confrontiamo i due periodi.

PT_START = "2026-03-19"

pre_pt_mask = dates_str < PT_START
post_pt_mask = dates_str >= PT_START

pre_vol = volumes[pre_pt_mask]
post_vol = volumes[post_pt_mask]

print(f"\n--- Confronto pre/post PT ({PT_START}) ---")
print(
    f"Set pre-PT:   {pre_pt_mask.sum():4d}  |  volume medio: {pre_vol.mean():6.1f} kg"
)
print(
    f"Set post-PT:  {post_pt_mask.sum():4d}  |  volume medio: {post_vol.mean():6.1f} kg"
)
print(f"Δ volume medio: {post_vol.mean() - pre_vol.mean():+.1f} kg per set")

# Bench press pre/post PT
bench_pre = bench_weights[bench_dates < PT_START]
bench_post = bench_weights[bench_dates >= PT_START]
if len(bench_pre) > 0 and len(bench_post) > 0:
    print(
        f"\nBench Press pre-PT:  media {bench_pre.mean():.1f} kg  (n={len(bench_pre)})"
    )
    print(
        f"Bench Press post-PT: media {bench_post.mean():.1f} kg  (n={len(bench_post)})"
    )


# ===========================================================================
# 9. PRODOTTO MATRICIALE — PESI COMBINATI
# ===========================================================================
# In ML un layer fully-connected calcola: output = X @ W + b
# Qui usiamo lo stesso meccanismo per combinare le feature con pesi arbitrari.
# Es: score = 0.5*volume + 0.3*reps + 0.2*weight (punteggio di intensità)

W_intensity = np.array([0.5, 0.3, 0.2])  # pesi per [volume, reps, weight_kg]

# Broadcasting: ogni riga di X_norm moltiplicata per W, poi sommata
intensity_scores = X_norm @ W_intensity  # (3628, 3) @ (3,) → (3628,)

print("\nIntensity scores (0–1):")
print(f"  media: {intensity_scores.mean():.3f}")
print(f"  max:   {intensity_scores.max():.3f} (set più intenso)")
print(f"  min:   {intensity_scores.min():.3f}")

# Esercizio con score più alto
top_idx = intensity_scores.argmax()
print(
    f"  Set più intenso: {exercise_names[top_idx]}, "
    f"{weights[top_idx]}kg × {reps[top_idx]} reps"
)


# ===========================================================================
# 10. RANDOM SEED — CAMPIONAMENTO RIPRODUCIBILE
# ===========================================================================
# In ML il campionamento riproducibile è fondamentale per train/test split,
# cross-validation, shuffle dei batch.

rng = np.random.default_rng(seed=42)

# Campiona 10 set casuali dal dataset
n = len(strength_rows)
sample_idx = rng.choice(n, size=10, replace=False)
sample_exercises = exercise_names[sample_idx]
sample_volumes = volumes[sample_idx]

print("\n10 set campionati casualmente (seed=42):")
for ex, vol in zip(sample_exercises, sample_volumes):
    print(f"  {ex:35s} volume: {vol:6.0f} kg")


# ===========================================================================
# 11. BENCHMARK
# ===========================================================================
# timeit gira in un namespace isolato, quindi le import vanno ripetute
# dentro la stringa di setup (non è un duplicato di quelle in cima al file).

setup = f"""
import csv
import numpy as np
with open('{WORKOUT_CSV}') as f:
    rows = list(csv.DictReader(f))
strength = [r for r in rows if float(r['Weight(kg)']) > 0 and int(r['Reps']) > 0]
w = [float(r['Weight(kg)']) for r in strength]
r = [int(r['Reps']) for r in strength]
wa = np.array(w); ra = np.array(r)
"""

t_list = timeit.timeit("[x*y for x,y in zip(w,r)]", setup=setup, number=1000)
t_numpy = timeit.timeit("wa * ra", setup=setup, number=1000)

print(f"\nBenchmark — calcolo volume ({len(strength_rows)} set, 1000 ripetizioni):")
print(f"  Lista:  {t_list:.4f}s")
print(f"  NumPy:  {t_numpy:.4f}s")
print(f"  Speedup: {t_list/t_numpy:.0f}x")
