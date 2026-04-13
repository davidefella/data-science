"""
Real dataset: workout data exported and updated from Fitbod (March 2025 → April 2026).
Last update: April 2026 (229 sessions, 3941 logged sets, 86 unique exercises)
Focus: core NumPy operations on real data, with an eye on ML/AI.
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Path to the dataset, derived from this file's location.
# .parents[2] goes up 2 directories: /01-NumPY → /10-python-ds-fundamentals → /data-science
# Note: the path is just a pointer to the file — it doesn't know or care
# about the format. If the file were .xlsx, the path would stay the same.
WORKOUT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "00-datasets" / "workoutExport.csv"
)

# 0. LOADING DATA
#
# Load the entire CSV into memory as a list of dicts (one dict per row).
# All values are strings.
# workouts_data_all → [
#   {"Date": "2025-03-15 ...","Exercise":"Barbell Bench Press","Weight(kg)":"80","Reps": "10",...},
#   {...},
#   {...},
#   ...
#   {...}]
with open(WORKOUT_DATA_PATH, encoding="utf-8") as workout_file:
    # DictReader returns an iterator (one-shot); list() loads all rows into memory
    workouts_data_all = list(csv.DictReader(workout_file))

print(f"Total row from csv file:        {len(workouts_data_all)}")
print(f"CSV columns:       {list(workouts_data_all[0].keys())}")
print(f"First row (sanity check): {workouts_data_all[0]}")


# ===========================================================================
# 1. WHY NUMPY — THE PROBLEM WITH PLAIN PYTHON LISTS
# ===========================================================================

# We want to compute the total volume (kg × reps) for each set.
# Each "row" is one dict from workouts_data_all, e.g.:
# row → {"Date": "2025-03-15 ...", "Exercise": "Barbell Bench Press", "Weight(kg)": "80", ...}
# volumes_list_py → [800.0, 360.0, 500.0, 720.0, ...]  (one number per set)
volumes_list_py = [
    float(row["Weight(kg)"]) * int(row["Reps"])
    for row in workouts_data_all
    if float(row["Weight(kg)"]) >= 0 and int(row["Reps"]) > 0
]
print(f"Sets with weight and reps: {len(volumes_list_py)}")
print(f"Total volume (list): {sum(volumes_list_py):,.0f} kg")

# Now let's do the same volume calculation with NumPy.

# Question: "Which rows are valid strength sets?"
# volumes_list_py → [800.0, 360.0, 500.0, ...]          (just numbers)
# valid_workout_rows → [{"Date":..., "Weight(kg)":"80", ...}, {...}, ...]  (full rows)
valid_workout_rows = [
    row
    for row in workouts_data_all
    if float(row["Weight(kg)"]) >= 0 and int(row["Reps"]) > 0
]

#   weights → [80.0, 45.0, 100.0, 60.0, ...]  (kg lifted per set)
#   reps    → [10,    8,    5,    12,   ...]   (reps per set)
weights = np.array([float(row["Weight(kg)"]) for row in valid_workout_rows])
reps = np.array([int(row["Reps"]) for row in valid_workout_rows])

# Element-wise multiplication: weights[0]*reps[0], weights[1]*reps[1], ...
# "In C" means NumPy runs this loop in compiled C code, not in Python.
# It operates on contiguous memory blocks without the Python interpreter overhead,
# which is why it's 10x–100x faster (see benchmark in section 11).
#
# How do you know it runs in C? You don't see it directly — it's an
# implementation detail of NumPy. Any operation between two np.arrays
# (like +, *, /, **) is automatically dispatched to compiled C code.
# That's the whole point of NumPy: you write Python, it runs C.
# Result preview:
#   weights = [80, 45, 100, ...]   ×   reps = [10, 8, 5, ...]
#   volumes_arr_np = [800, 360, 500, ...]  (kg × reps for each set)
volumes_arr_np = weights * reps
print(f"Total volume (NumPy): {volumes_arr_np.sum():,.0f} kg")

# Benchmark: on a small dataset the difference is minimal, but on millions of rows
# (e.g. workout logs for an entire platform) NumPy is 10x–100x faster.


# ===========================================================================
# 2. FUNDAMENTAL ATTRIBUTES
# ===========================================================================
# Every np.array has 3 key attributes — like an ID card:
#   shape  → dimensions: (3628,) = 1D vector, (3628, 3) = 2D table
#   dtype  → data type of EVERY element (all must be the same type!)
#   nbytes → memory used in bytes = num_elements × bytes_per_element
#
# Unlike Python lists, an np.array can only hold ONE type. Common dtypes:
#   float64 / float32  → decimals (default is float64, ML often uses float32)
#   int64 / int32      → integers
#   bool               → True/False (used for masks)
#   U10, U50...        → fixed-length strings (e.g. exercise_names)
# If you mix types, NumPy auto-converts to the most general one
# (e.g. ints + floats → all floats, numbers + strings → all strings).

print(f"\nweights shape:  {weights.shape}")  # (3628,) — 1D vector
print(f"weights dtype:  {weights.dtype}")  # float64
print(f"weights nbytes: {weights.nbytes} B")  # 3628 * 8 = 29024 bytes


# column_stack takes separate arrays and puts them side by side as columns
# of a single table (matrix). Think of it like building a spreadsheet:
#
#   weights = [80, 60, 70, ...]    (column 1)
#   reps    = [10,  8, 12, ...]    (column 2)
#   volumes_arr_np = [800, 480, 840, ...] (column 3)
#
#   column_stack → | 80  10  800 |   ← set 1
#                  | 60   8  480 |   ← set 2
#                  | 70  12  840 |   ← set 3
#
# In ML this is called a "feature matrix": each row is one sample (a set),
# each column is one feature (weight, reps, volume).

# Note: since this is a np.array, all values must be the same type (float64).
# All arrays must have the same length, otherwise → ValueError.
# For mixed types (strings + numbers), use pandas DataFrame instead.

# Why "X"? In ML the standard equation is: y = f(X)
#   X → input data (features: what you feed to the model)
#   y → output data (target: what you want to predict)
X = np.column_stack([weights, reps, volumes_arr_np])
print(f"\nFeature matrix X shape: {X.shape}")  # (3628, 3)
print("Columns: [weight_kg, reps, volume]")
print(f"First 5 rows:\n{X[:5]}")


# ===========================================================================
# 3. DTYPE AND MEMORY
# ===========================================================================
# When NumPy encounters decimals, it uses float64 (8 bytes per number).
#   np.array([1, 2, 3])     → int64   (all integers → stays int)
#   np.array([1, 2, 3.0])   → float64 (at least one decimal → all become float)

# astype() converts the array to a different data type.
# Here: float64 (8 bytes per number) → float32 (4 bytes per number).
# Same values, half the memory. Precision loss is negligible for ML.
X_f32 = X.astype(np.float32)


# ===========================================================================
# 4. DESCRIPTIVE STATISTICS — AXIS
# ===========================================================================

# "features" = the characteristics that describe each sample.
# Same concept as "columns" in a database or "variables" in statistics.
print("\n--- Statistics per feature (axis=0) ---")
feature_names = ["weight_kg", "reps", "volume"]

# Axis — the direction along which you aggregate:
#
#            weight  reps  volume
#   set 1  |  80     10    800  | → axis=1 →  (across columns, one result per row)
#   set 2  |  60      8    480  | →
#   set 3  |  70     12    840  | →
#              ↓      ↓     ↓
#           axis=0
#     (across rows, one result per column)
#
#   axis=0 → "for each column, aggregate all the rows"
#            X.mean(axis=0) on the 3 rows above:
#            [(80+60+70)/3, (10+8+12)/3, (800+480+840)/3] → [70.0, 10.0, 706.7]

#   axis=1 → "for each row, aggregate all the columns"
#            X.mean(axis=1) on the 3 rows above:
#            [(80+10+800)/3, (60+8+480)/3, (70+12+840)/3] → [296.7, 182.7, 307.3]

# NumPy supports up to 32 dimensions (axis=0, 1, 2, ...):
#   1D → vector (list of numbers)
#   2D → table (our case: sets × features)
#   3D → images, time series (batch × timestep × feature)


# enumerate() gives back pairs of (index, value) at each iteration:
#   iteration 1 → i=0, name="weight_kg"
#   iteration 2 → i=1, name="reps"
#   iteration 3 → i=2, name="volume"
for i, name in enumerate(feature_names):
    # X[:, i] uses two coordinates: [rows, column]
    #   :  → "all rows" (from first to last)
    #   i  → "only column number i"
    # Like selecting an entire column in Excel by clicking the letter.
    #
    #   X[:, 0] → [80, 60, 70, ...]  (all weights)
    #   X[:, 1] → [10,  8, 12, ...]  (all reps)
    #   X[:, 2] → [800, 480, 840, ...] (all volumes)

    # col is a 1D np.array — one entire column from X.
    # e.g. when i=0: col → [80.0, 60.0, 70.0, 55.0, ...]  (all 3628 weights)
    col = X[:, i]

    # Raw data (first 5 values) before aggregation:
    print(f"  {name:12s}  raw (first 5): {col[:5]}")

    # Now aggregate — each function reduces the whole array to one number.
    # Each one answers a different question:
    #   mean → "What's the typical value?"
    #   std  → "How much do values vary?"
    #   min  → "What's the lowest value?"
    #   max  → "What's the highest value?"
    col_mean = col.mean()
    col_std = col.std()
    col_min = col.min()
    col_max = col.max()
    print(
        f"  {name:12s}  mean={col_mean:.2f}  std={col_std:.2f}  "
        f"min={col_min:.1f}  max={col_max:.1f}"
    )


# ===========================================================================
# 5. INDEXING AND BOOLEAN INDEXING
# ===========================================================================
# Extract all Barbell Bench Press sets and analyze the progression.

# Build arrays of exercise names and dates from the filtered rows.
# Preview:
#   exercise_names → ["Barbell Bench Press", "Lat Pulldown", "Squat", ...]
#   dates_str      → ["2025-03-15", "2025-03-15", "2025-03-17", ...]
exercise_names = np.array([row["Exercise"] for row in valid_workout_rows])

# row["Date"] is something like "2025-03-15 10:30:00" — [:10] takes
# only the first 10 chars → "2025-03-15" (just the date, no time).
# It's not printed to terminal, it's just stored for later filtering.
dates_str = np.array([row["Date"][:10] for row in valid_workout_rows])

# Boolean indexing — the core filtering technique in NumPy.
# Step by step:
#   1. exercise_names == "Barbell Bench Press" compares EVERY element
#      and returns a new array with same length, dtype=bool:
#      bench_mask → [True, False, False, True, False, ...]
#      (In plain Python, list == "value" doesn't work this way — you'd need a loop.
#      NumPy overloads == to compare each element automatically, in C.)
#
#   2. weights[bench_mask] — positional filtering.
#      Mask and array have the same length. NumPy overlaps them
#      position by position: True → keep, False → discard.
#
#      position:   0     1      2     3     4
#      weights:  [80,   45,   100,   85,   60]
#      mask:     [ T,    F,     F,    T,    F]
#      result:   [80,                85      ]  → [80, 85]
#
#      (NumPy-only feature. In plain Python, list[list_of_bools]
#      raises an error — you'd need: [w for w, m in zip(weights, mask) if m])
#
#   3. bench_mask.sum() counts the True values (True=1, False=0)
bench_mask = exercise_names == "Barbell Bench Press"

# Reminder: weights contains ALL exercises (3628 sets).
# bench_mask filters only the "Barbell Bench Press" ones.
# bb_bench_weights → [80.0, 82.5, 80.0, 85.0, ...]  (only bench press weights)
bb_bench_weights = weights[bench_mask]
bench_dates = dates_str[bench_mask]

# Question: "How many bench press sets, and at what weights?"
# .sum() on mask → "How many sets?"
print(f"\nBarbell Bench Press: {bench_mask.sum()} sets")

# .min()/.max()  → "What's the weight range?""
print(f"Min weight: {bb_bench_weights.min()} kg  |  max: {bb_bench_weights.max()} kg")

# .mean() → "What do I lift on average?"
# .std()  → "Am I consistent?"
print(
    f"Mean: {bb_bench_weights.mean():.1f} kg  |  std: {bb_bench_weights.std():.1f} kg"
)

# Same for Lat Pulldown
lat_mask = exercise_names == "Lat Pulldown"
# Reminder — weights looks like this:
#   weights → [80.0, 45.0, 100.0, 60.0, 72.5, ...]  (one per set)
# weights[lat_mask] keeps only the weights where lat_mask is True.
lat_pulldown_weights = weights[lat_mask]
print(f"\nLat Pulldown: {lat_mask.sum()} sets")
print(f"Min weight: {lat_pulldown_weights.min()} kg  |  max: {lat_pulldown_weights.max()} kg")
print(f"Mean:       {lat_pulldown_weights.mean():.1f} kg  |  std: {lat_pulldown_weights.std():.1f} kg")


# ===========================================================================
# 6. RESHAPE AND BROADCASTING — NORMALIZATION
# ===========================================================================
# Before feeding data to an ML model, you normalize the features.
# With NumPy this is done via broadcasting — no loops needed.

# Min-Max scaling
# Question: "Where does this value sit between my min and max?"
# Domanda: "Quanto è alto questo valore rispetto al mio minimo e massimo?"
# Scales everything to 0–1. If max weight is 140kg and you lift 70kg → 0.5.
# Why? An ML algorithm would see "140 kg" and "12 reps" as very different
# numbers, but both are high values in their range. Normalization makes
# features comparable.
#
# Reminder — X is the feature matrix built in section 2:
#   X → | 80.0  10  800.0 |   ← set 1 (weight, reps, volume)
#       | 60.0   8  480.0 |   ← set 2
#       | 70.0  12  840.0 |   ← set 3
#       | ...  ...  ...   |   (3628 rows × 3 columns)
#
# "shape" is the dimensions of the array:
#   X.shape       → (3628, 3)  — 3628 rows, 3 columns
#   weights.shape → (3628,)    — 1D array with 3628 elements
#   X_min.shape   → (3,)       — 1D array with 3 elements (one min per column)

X_min = X.min(axis=0)  # → e.g. [0.0,  1,    0.0]  (min of each column)
X_max = X.max(axis=0)  # → e.g. [140.0, 30, 4200.0] (max of each column)

# Broadcasting: NumPy automatically "stretches" smaller arrays to match
# the bigger one. Here X is (3628, 3) and X_min is (3,).
# NumPy applies X_min to EVERY row automatically:
#   row 0: [80, 10, 800] - [0, 1, 0] = [80, 9, 800]
#   row 1: [60,  8, 480] - [0, 1, 0] = [60, 7, 480]
#   ... same subtraction for all 3628 rows, no loop needed.
X_norm = (X - X_min) / (X_max - X_min)

print("\nAfter Min-Max scaling:")
print(f"  min per column: {X_norm.min(axis=0)}")
print(f"  max per column: {X_norm.max(axis=0)}")

# Z-score standardization
# Question: "How far is this value from the average?"
# Domanda: "Quanto è lontano questo valore dalla media?"
# The answer is in units of standard deviation:
#   z = +1.0 → "1 std above average" / "1 std sopra la media"
#   z =  0.0 → "Exactly average" / "Esattamente nella media"
#   z = -2.0 → "2 std below average" / "2 std sotto la media"
#
# Formula: z = (value - mean) / std
# Example: if avg weight is 50kg and std is 20kg,
#   a set at 70kg → z = (70 - 50) / 20 = +1.0  (above average)
#   a set at 30kg → z = (30 - 50) / 20 = -1.0  (below average)
#
# Difference with Min-Max: Min-Max tells you "where are you between
# min and max", Z-score tells you "how far are you from typical
# behavior". Both make features comparable for ML.
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# X_std_scaled: the entire feature matrix X, but z-score normalized.
# Each value now tells you "how many standard deviations from the mean".
X_std_scaled = (X - X_mean) / X_std

print("\nAfter Z-score:")
print(f"  mean per column: {X_std_scaled.mean(axis=0).round(6)}")
print(f"  std per column:  {X_std_scaled.std(axis=0).round(6)}")


# ===========================================================================
# 7. WEEKLY AGGREGATIONS — VOLUME OVER TIME
# ===========================================================================
# Question: "How does my training volume change week by week?"
# Useful to see if you're progressively overloading or tapering off.

# Group by week with Python, then use NumPy for the calculations
# defaultdict extends dict — same behavior, plus one extra:
# if you access a key that doesn't exist, it auto-creates it with a default value.
#   defaultdict(float) → missing keys start at 0.0
#   defaultdict(int)   → missing keys start at 0
#   defaultdict(list)  → missing keys start at []

# So weekly_volumes["2025-W12"] += 500 works even the first time,
# without needing to check if the key exists first.
weekly_volumes = defaultdict(float)
weekly_sets = defaultdict(int)

for row in valid_workout_rows:
    # "2025-03-15" → datetime object
    d = datetime.strptime(row["Date"][:10], "%Y-%m-%d")

    # datetime → "2025-W11" (year + week number)
    # strftime = "string format time" — Python built-in, not NumPy.
    # %Y=year, %W=week number (00-53), %m=month, %d=day
    week = d.strftime("%Y-W%W")

    # volume for this set
    vol = float(row["Weight(kg)"]) * int(row["Reps"])
    
    # accumulate into the week's bucket (defaultdict auto-creates the key)
    weekly_volumes[week] += vol
    weekly_sets[week] += 1

# After the loop:
# weekly_volumes → {"2025-W10": 4500.0, "2025-W11": 3200.0, "2025-W12": 5100.0, ...}
# weekly_sets    → {"2025-W10": 25,      "2025-W11": 18,      "2025-W12": 30, ...}

# weeks → ["2025-W10", "2025-W11", "2025-W12", ...]  (sorted chronologically)
weeks = sorted(weekly_volumes.keys())

# vol_array  → [4500.0, 3200.0, 5100.0, ...]  (total kg per week)
vol_array = np.array([weekly_volumes[w] for w in weeks])

# sets_array → [25, 18, 30, ...]               (number of sets per week)
sets_array = np.array([weekly_sets[w] for w in weeks])

print(f"\nTotal weeks: {len(weeks)}")

# mean → "What's a typical week?
print(f"Average weekly volume: {vol_array.mean():,.0f} kg")

# argmax/argmin return the INDEX (position) of the max/min value, not the value itself.
# e.g. vol_array = [4500, 3200, 5100, 2800]
#      argmax() → 2  (position of 5100), argmin() → 3  (position of 2800)
# Then weeks[2] gives the corresponding week name.
print(f"Max volume:  week {weeks[vol_array.argmax()]} — {vol_array.max():,.0f} kg")
print(f"Min volume:  week {weeks[vol_array.argmin()]} — {vol_array.min():,.0f} kg")
print(f"Average sets/week: {sets_array.mean():.1f}")


# ===========================================================================
# 8. PERIOD COMPARISON — PRE/POST PT
# ===========================================================================
# Question: "Since getting a PT, am I training harder?"
# Compare average volume before and after January 8, 2026.

PT_START = "2026-01-08"

# Same boolean mask technique as section 5, but comparing dates (strings).
# Works because "YYYY-MM-DD" sorts chronologically as text.

# pre_pt_mask  → [True, True, ..., False, False]  (before PT)
# post_pt_mask → [False, False, ..., True, True]  (after PT)
pre_pt_mask = dates_str < PT_START
post_pt_mask = dates_str >= PT_START

pre_pt_vol = volumes_arr_np[pre_pt_mask]
post_pt_vol = volumes_arr_np[post_pt_mask]

print(f"\n--- Pre/post PT comparison ({PT_START}) ---")
print(f"Pre-PT sets:   {pre_pt_mask.sum():4d}  |  avg volume: {pre_pt_vol.mean():6.1f} kg")
print(
    f"Post-PT sets:  {post_pt_mask.sum():4d}  |  avg volume: {post_pt_vol.mean():6.1f} kg"
)

# Δ (delta) → "By how much did avg volume change?"
# Positive = improved, negative = worsened
print(f"Δ avg volume: {post_pt_vol.mean() - pre_pt_vol.mean():+.1f} kg per set")


# ===========================================================================
# 9. MATRIX PRODUCT — COMBINED WEIGHTS
# ===========================================================================
# Question: "What was my single most intense set ever?"
# To answer, we need a single score that combines weight, reps and volume.
#
# This is a WEIGHTED AVERAGE — each feature gets a different importance:
#   score = 0.5 * volume  +  0.3 * reps  +  0.2 * weight
#           (50% weight)     (30% weight)   (20% weight)
#
# Example for one set (using normalized 0-1 values from X_norm):
#   X_norm row = [0.57, 0.33, 0.19]  (normalized weight, reps, volume)
#   score = 0.5*0.19  +  0.3*0.33  +  0.2*0.57
#         = 0.095     +  0.099     +  0.114
#         = 0.308
#
# NOTE: we use X_norm (normalized 0-1), not raw X. Otherwise volume
# (0-4000+) would dominate reps (1-30) just because of its larger scale.
#
# The weights (0.5, 0.3, 0.2) are ARBITRARY — we chose them manually.
# In ML, the model LEARNS these weights during training. Here we fix
# them by hand just to show the mechanism.
#
# The @ operator does this multiply+sum for ALL 3628 rows at once,
# no loop needed. This is the same operation a neuron does: output = X @ W + b

W_intensity = np.array([0.5, 0.3, 0.2])  # importance of [weight, reps, volume]

# Reminder: X_norm is the feature matrix X after Min-Max normalization (section 6).
# All values are between 0 and 1, so features are comparable.
# X_norm → | 0.57  0.33  0.19 |   ← set 1 (normalized weight, reps, volume)
#          | 0.43  0.27  0.11 |   ← set 2
#          | 0.50  0.40  0.20 |   ← set 3
#          | ...   ...   ...  |   (3628 rows × 3 columns)
# @ = matrix-vector product: for each row, multiply element-wise by W, then sum.
#
# (3628, 3)  @  (3,)        →  (3628,)
#  ↑              ↑               ↑
#  X_norm         W_intensity     intensity_scores
#  3628 rows      3 elements      3628 scores
#  3 columns      (one per        (one per set)
#  (weight,       column)
#   reps,
#   volume)
#
# The "3" must match: 3 columns in X_norm × 3 elements in W_intensity.
# If they don't match → ValueError.
intensity_scores = X_norm @ W_intensity

print("\nIntensity scores (0–1):")
print(f"  mean: {intensity_scores.mean():.3f}")
print(f"  max:  {intensity_scores.max():.3f} (most intense set)")
print(f"  min:  {intensity_scores.min():.3f}")

# Exercise with the highest score
top_idx = intensity_scores.argmax()
print(
    f"  Most intense set: {exercise_names[top_idx]}, "
    f"{weights[top_idx]}kg × {reps[top_idx]} reps"
)


# ===========================================================================
# 10. RANDOM SEED — REPRODUCIBLE SAMPLING
# ===========================================================================
# In ML you need randomness (to split data, shuffle batches, etc.)
# but you also need REPRODUCIBILITY — same random results every time
# you run the script. That's what the seed does.
#
# seed = a starting point for the random number generator.
# Same seed → same "random" numbers every time.
# Different seed → different numbers.
# Without seed → different results on every run (not reproducible).

rng = np.random.default_rng(seed=42)  # 42 is arbitrary, any number works

# rng.choice(n, size=10, replace=False) picks 10 random indices from 0 to n-1.
#   n       → how many elements to choose from (3628 sets)
#   size=10 → how many to pick
#   replace=False → no duplicates (can't pick the same set twice)

# sample_idx → e.g. [2847, 105, 3201, 742, ...]  (10 random positions)
n = len(valid_workout_rows)
sample_idx = rng.choice(n, size=10, replace=False)

# Then use those indices to grab the corresponding exercises and volumes
sample_exercises = exercise_names[sample_idx]
sample_volumes = volumes_arr_np[sample_idx]

print("\n10 randomly sampled sets (seed=42):")
for ex, vol in zip(sample_exercises, sample_volumes):
    print(f"  {ex:35s} volume: {vol:6.0f} kg")


