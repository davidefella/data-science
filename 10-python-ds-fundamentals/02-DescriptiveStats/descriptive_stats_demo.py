"""
Descriptive Statistics on real personal datasets.
Datasets: wallet (expenses), health (steps/weight/sleep), investments.
Focus: measures of central tendency and dispersion, with Python code.
"""

import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gmean, iqr, trim_mean

# Add parent directory to path so we can import the shared data_loader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# pylint: disable=import-error,wrong-import-position
from data_loader import load_csv  # noqa: E402
# pylint: enable=import-error,wrong-import-position

# ===========================================================================
# 0. LOADING DATA
# ===========================================================================
# load_csv() is a shared utility (see ../data_loader.py).
# It reads a CSV from 00-datasets/ and returns a list of dicts (one per row).
# All values are strings — numeric conversion happens later.

# wallet_data → [{"date": "2026-03-31", "category": "Canone Intesa",
#                 "macro_category": "BANCA", "amount": "-7.5", ...}, ...]
wallet_data = load_csv("wallet.csv")

# health_data → [{"date": "2021-06-24", "steps": "", "weight_kg": "77.9",
#                 "sleep_total_hr": ""}, ...]
health_data = load_csv("health.csv")

# investments_data → [{"Data": "2026-03-27", "Categoria": "Fineco ETF iShare MSCI World",
#                      "Importo": "-107.55", "Valuta": "EUR"}, ...]
investments_data = load_csv("investments.csv")

print(f"Wallet rows:      {len(wallet_data)}")
print(f"Health rows:      {len(health_data)}")
print(f"Investments rows: {len(investments_data)}")
print(f"Wallet first row (sanity check): {wallet_data[0]}")


# ===========================================================================
# 1. EXTRACT NUMERIC ARRAYS
# ===========================================================================
# CSV values are strings — we convert to float and filter out empty/invalid values.

# All expense amounts (negative = spending, positive = income).
# We take the absolute value so we work with "how much was spent".
# Each row is one dict from wallet_data, e.g.:
#   row → {"date": "2026-03-31", "category": "Canone Intesa", "amount": "-7.5", ...}
# expenses_all → [7.5, 90.8, 10.0, 20.0, ...]  (one per transaction)
expenses_all = np.array(
    [
        abs(float(row["amount"]))
        for row in wallet_data
        if row["amount"] and float(row["amount"]) < 0  # only expenses (negative)
    ]
)

# Categories for each expense — one entry per TRANSACTION, not per unique category.
# "BANCA" appears many times because there are many bank transactions.
# Aligned positionally with expenses_all:
#   expenses_all       → [7.5,     90.8,       10.0,      3.95,    15.0,         7.5,    ...]
#   expense_categories_all → ["BANCA", "TRASPORTO", "BANCA", "FOOD & DRINK", "BANCA", ...]
#   Position 0 = 7.5 EUR in BANCA, position 1 = 90.8 EUR in TRASPORTO, etc.
expense_categories_all = np.array(
    [
        row["macro_category"]
        for row in wallet_data
        if row["amount"] and float(row["amount"]) < 0
    ]
)

# Micro categories (e.g. "Benzina", "Canone Intesa San Paolo", "Svago & Uscite")
# Same positional alignment as expenses_all and expense_categories_all.
expense_micro_categories_all = np.array(
    [
        row["category"]
        for row in wallet_data
        if row["amount"] and float(row["amount"]) < 0
    ]
)

# Each row is one dict from health_data, e.g.:
#   row → {"date": "2025-01-01", "steps": "12728", "weight_kg": "79.3", "sleep_total_hr": "7.24"}
# daily_steps_all → [7756.0, 12345.0, 8900.0, ...]
daily_steps_all = np.array(
    [
        float(row["steps"])
        for row in health_data
        if row["steps"]  # filter out empty values
    ]
)

# Body weight over time — filter out empty values
# body_weight_all → [77.9, 77.7, 77.5, ...]
body_weight_all = np.array(
    [float(row["weight_kg"]) for row in health_data if row["weight_kg"]]
)

# Sleep hours — filter out empty values
# sleep_hours_all → [7.24, 6.54, ...]
sleep_hours_all = np.array(
    [float(row["sleep_total_hr"]) for row in health_data if row["sleep_total_hr"]]
)

# Date ranges and day counts for context
wallet_dates = sorted({row["date"] for row in wallet_data if row["date"]})
health_dates = sorted({row["date"] for row in health_data if row["date"]})
invest_dates = sorted({row["Data"] for row in investments_data if row["Data"]})

print(
    f"\nExpenses:    {len(expenses_all)} transactions  "
    f"({wallet_dates[0]} → {wallet_dates[-1]}, {len(wallet_dates)} unique days)"
)
print(
    f"Investments: {len(investments_data)} transactions  "
    f"({invest_dates[0]} → {invest_dates[-1]}, {len(invest_dates)} unique days)"
)
print(
    f"Daily steps: {len(daily_steps_all)} days  "
    f"({health_dates[0]} → {health_dates[-1]}, {len(health_dates)} unique days)"
)
print(f"Body weight: {len(body_weight_all)} measurements  " f"(same health dataset)")
print(f"Sleep hours: {len(sleep_hours_all)} nights  " f"(same health dataset)")


print(f"\nExpenses (first 5): {expenses_all[:5]}")


# ===========================================================================
# 2. ARITHMETIC MEAN — statistics.mean() / np.mean()
# ===========================================================================
# Question: "How much do I spend on average per transaction?"
#
# Formula: x_bar = sum(all values) / count
#
# The mean sums everything and divides by n.
# Problem: OUTLIERS drag the mean. One big expense (e.g. rent €800)
# pulls the average up, making it unrepresentative of typical spending.

mean_expenses = np.mean(expenses_all)
mean_expenses_std = statistics.mean(expenses_all)  # same result, stdlib version

mean_steps = np.mean(daily_steps_all)
mean_weight = np.mean(body_weight_all)
mean_sleep = np.mean(sleep_hours_all)

print("\n" + "=" * 60)
print("  2. ARITHMETIC MEAN")
print("=" * 60)
print(f"Mean expense (np):         {mean_expenses:.2f} EUR")
print(f"Mean daily steps:          {mean_steps:.0f}")
print(f"Mean body weight:          {mean_weight:.1f} kg")
print(f"Mean sleep:                {mean_sleep:.2f} h")


# argsort() returns the INDICES that would sort the array (not the values).
# Example: expenses = [10, 50, 3, 80] → argsort → [2, 0, 1, 3]
#   meaning: smallest is at position 2 (3€), then 0 (10€), then 1 (50€), then 3 (80€)
# We need indices (not values) so we can also grab the matching category.
top_5_idx = np.argsort(expenses_all)[::-1][:5]
print("Top 5 expenses:")
for idx in top_5_idx:
    print(f"  {expenses_all[idx]:8.2f} EUR — {expense_micro_categories_all[idx]}")
print("  -> These outliers pull the mean UP, away from 'typical' spending")


# ===========================================================================
# 3. MEDIAN — statistics.median() / np.median()
# ===========================================================================
# Question: "What is the expense that sits exactly in the middle?"
# The CENTRAL value when data is sorted.
# If n is even, it's the average of the two middle values.
#
#   sorted: [3, 7, 10, 15, 100]
#                   ↑ median = 10 (middle position)

median_expenses = np.median(expenses_all)

print("\n" + "=" * 60)
print("  3. MEDIAN")
print("=" * 60)
print(f"Median expense: {median_expenses:.2f} EUR")
print(f"Mean expense:   {mean_expenses:.2f} EUR")
print(f"Difference:     {mean_expenses - median_expenses:.2f} EUR")
# A simple way to check skewness: mean/median ratio.
#   ≈ 1.0 → symmetric, mean and median agree
#   > 1.5 → significant skew
#   > 2.0 → strong skew (outliers pulling the mean away)
mean_median_ratio = mean_expenses / median_expenses
print(f"Mean/Median ratio: {mean_median_ratio:.1f}x")
print(f"  -> Ratio {mean_median_ratio:.1f}x = strongly skewed, median is more reliable")

# Practical rule: always compute BOTH mean and median.
# If they differ a lot, the distribution is asymmetric and the median
# is more reliable as "typical value".


# ===========================================================================
# 4. TRIMMED MEAN — scipy.stats.trim_mean()
# ===========================================================================
# Question: "What is the average if I ignore the extreme values?"
#
# Removes a fixed percentage from BOTH ends (top and bottom),
# then computes the mean on the rest.
# Useful when outliers are systematic (not measurement errors)
# and you still want a mean-like measure.
#
# How it works with proportiontocut=0.1 (10%):
#   1. Sort the data
#   2. Remove the bottom 10% AND the top 10%
#   3. Compute the mean on the remaining 80%
#
# Note: trim_mean is NOT in the statistics stdlib — it requires scipy.

trimmed_mean_10 = trim_mean(expenses_all, proportiontocut=0.1)
trimmed_mean_20 = trim_mean(expenses_all, proportiontocut=0.2)

print("\n" + "=" * 60)
print("  4. TRIMMED MEAN")
print("=" * 60)
print(f"Mean (full):     {mean_expenses:.2f} EUR")
print(f"Trimmed 10%:     {trimmed_mean_10:.2f} EUR")
print(f"Trimmed 20%:     {trimmed_mean_20:.2f} EUR")
print(f"Median:          {median_expenses:.2f} EUR")
print("  -> Trimmed mean falls between mean and median")


# ===========================================================================
# 5. WEIGHTED MEAN — np.average(data, weights=w)
# ===========================================================================
# Question: "What is my average daily steps, giving more weight to recent days?"
#
# Each value gets multiplied by its weight, then you divide by sum of weights.
# Formula: x_w = sum(x_i * w_i) / sum(w_i)
#
# Values with higher weight "pull" the average more.
# Unlike the arithmetic mean where all values count equally.
#
# Note: statistics.mean() does NOT support weights — use np.average().

# Give more weight to recent days: weight = 1, 2, 3, ..., n
# np.arange(1, n+1) works like range() but returns a NumPy array.
# First day (oldest) has weight 1, last day (most recent) has weight n_days.
# So recent days count MUCH more in the weighted average.
n_days = len(daily_steps_all)
recency_weights = np.arange(1, n_days + 1)
# recency_weights → [1, 2, 3, ..., 1718]  (e.g. today's steps weigh 1718x more than day 1)

# Inputs:
#   daily_steps_all  → [7756, 12345, 8900, ..., 10200]   (1718 values)
#   recency_weights  → [   1,     2,    3, ...,  1718]   (1718 weights)
#
# simple_mean_steps → (7756 + 12345 + 8900 + ...) / 1718 = one number
#   Every day counts equally.
simple_mean_steps = np.mean(daily_steps_all)

# weighted_mean_steps → (7756×1 + 12345×2 + 8900×3 + ... + 10200×1718) / (1+2+3+...+1718)
#   Recent days pull the average more.
# Note: weights and data must have the same length (positional alignment).
# If len(weights) != len(data) → ValueError.
weighted_mean_steps = np.average(daily_steps_all, weights=recency_weights)

print("\n" + "=" * 60)
print("  5. WEIGHTED MEAN")
print("=" * 60)
print(f"Simple mean steps:   {simple_mean_steps:.0f}")
print(f"Weighted mean steps: {weighted_mean_steps:.0f}  (recent days count more)")
print("  -> If you're walking more recently, weighted > simple")
print("  -> If you're walking less recently, weighted < simple")


# ===========================================================================
# 6. GEOMETRIC MEAN — scipy.stats.gmean()
# ===========================================================================
# Question: "What is my average monthly investment return?"
# Domanda: "Qual è il mio rendimento medio mensile di investimento?"
#
# The nth root of the product of all values.
# Formula: G = (x1 * x2 * ... * xn) ^ (1/n)
#
# When to use: growth rates, compound returns, ratios.
# The arithmetic mean OVERESTIMATES compound growth.
#
# Example: an investment goes +50%, -20%, +30% over 3 years.
#   As factors: 1.50, 0.80, 1.30
#   Arithmetic mean of returns: (50 - 20 + 30) / 3 = 20% — WRONG (overestimates)
#   Geometric mean: (1.5 * 0.8 * 1.3)^(1/3) = 1.1696 → +16.96% — CORRECT

# Real data: monthly returns of iShares MSCI World ETF (SWDA.MI) Feb 2024 → Mar 2026.
# Source: Yahoo Finance via yfinance.
# These are PERCENTAGE returns: +3.71% means the ETF grew 3.71% that month.
monthly_returns_pct = np.array(
    [
        +3.71,
        -2.03,
        +1.24,
        +4.95,
        +0.20,
        -0.31,
        +1.39,
        +1.20,
        +7.47,
        -0.87,
        +4.10,
        -2.55,
        -7.89,
        -3.88,
        +6.35,
        +0.89,
        +4.77,
        -0.31,
        +2.51,
        +4.28,
        -0.35,
        +0.34,
        +0.34,
        +1.24,
        -4.81,
    ]
)

# Convert percentages to GROWTH FACTORS: factor = 1 + (pct / 100)
#   +3.71% → 1.0371  (you have 103.71% of what you had → growth)
#   -2.03% → 0.9797  (you have 97.97% of what you had → loss)
#    0.00% → 1.0000  (unchanged)
#
# Why factors? Compound returns work by MULTIPLYING, not adding:
#   100€ × 1.0371 × 0.9797 = 101.63€  (real result after 2 months)
#   Arithmetic mean would give: (3.71 - 2.03)/2 = +0.84%
#   → 100€ × 1.0084 × 1.0084 = 101.69€ (wrong, overestimates)
# monthly_factors → [1.0371, 0.9797, 1.0124, 1.0495, ...]  (25 values, all around 1)
#   > 1 = positive month, < 1 = negative month
monthly_factors = 1 + monthly_returns_pct / 100

# Arithmetic mean of returns — WRONG for compound growth
arith_mean_return = np.mean(monthly_returns_pct)

# Geometric mean of factors — CORRECT for compound growth
geo_mean_factor = gmean(monthly_factors)
geo_mean_return = (geo_mean_factor - 1) * 100

print("\n" + "=" * 60)
print("  6. GEOMETRIC MEAN")
print("=" * 60)
print(f"MSCI World ETF — {len(monthly_returns_pct)} months (Feb 2024 → Mar 2026)")
print(f"Arithmetic mean of returns: {arith_mean_return:+.2f}%/month  (overestimates!)")
print(f"Geometric mean of returns:  {geo_mean_return:+.2f}%/month  (correct)")
print(
    f"  -> Arithmetic says ~{arith_mean_return * 12:.1f}%/year, "
    f"geometric says ~{geo_mean_return * 12:.1f}%/year"
)


# ===========================================================================
# 7. HARMONIC MEAN — statistics.harmonic_mean()
# ===========================================================================
# Question: "What is my real average speed?"
#
# The reciprocal of the arithmetic mean of the reciprocals.
# Formula: H = n / sum(1/x_i)
#
# In plain words: invert every value, take the mean, invert back.
#
# When to use: rates, speeds, ratios (km/h, price/unit, F1-score).
# The arithmetic mean of speeds is WRONG because you spend more TIME
# at the slower speed.
#
# Example: 100km at 60 km/h, then 100km at 120 km/h.
#   Arithmetic mean: (60 + 120) / 2 = 90 km/h — WRONG
#   Real speed: 200km / (100/60 + 100/120)h = 200/2.5 = 80 km/h
#   Harmonic mean: 2 / (1/60 + 1/120) = 80 km/h — CORRECT

# Real data: walking speeds from workout dataset.
# Speed = distance(km) / duration(h) for each walking session.
# Filter out sessions below 2 km/h (likely GPS tracking errors or pauses).
walking_speeds_raw = np.array(
    [
        (float(row["Distance(m)"]) / 1000) / (float(row["Duration(s)"]) / 3600)
        for row in load_csv("workoutExport.csv")
        if row["Exercise"] == "Walking"
        and float(row["Distance(m)"]) > 0
        and float(row["Duration(s)"]) > 0
    ]
)

walking_speeds_valid = walking_speeds_raw[walking_speeds_raw >= 2.0]
# walking_speeds_valid → [3.8, 4.5, 4.5, 5.9, 3.3, ...]  (km/h per session)

arith_mean_speed = np.mean(walking_speeds_valid)
harmonic_mean_speed = statistics.harmonic_mean(walking_speeds_valid)

print("\n" + "=" * 60)
print("  7. HARMONIC MEAN")
print("=" * 60)
print(f"Walking sessions: {len(walking_speeds_valid)}")
print(f"Arithmetic mean speed: {arith_mean_speed:.2f} km/h  (overestimates)")
print(f"Harmonic mean speed:   {harmonic_mean_speed:.2f} km/h  (correct)")
print("  -> Harmonic is lower because slow walks take MORE TIME,")
print("     so they should weigh more in the average speed.")
print("  -> Harmonic < geometric < arithmetic (always, for positive values)")


# ===========================================================================
# 8. MODE — statistics.mode() / scipy.stats.mode()
# ===========================================================================
# Question: "What is my most common expense category?"
#
# The value that appears most frequently.
# Only measure that works on CATEGORICAL data (strings, labels).
# Also works on discrete numeric data with few distinct values.
# Useless on continuous data (every value appears once).
#
# Note: NumPy does NOT have a native mode function.
# For strings/categories → use statistics.mode() (scipy dropped string support in v1.11)
# For numeric data → scipy.stats.mode() also works and returns count.

mode_category = statistics.mode(expense_categories_all)

# Count occurrences of each category
unique_categories, category_counts = np.unique(
    expense_categories_all, return_counts=True
)
# Sort by count (descending)
sorted_idx = np.argsort(category_counts)[::-1]

print("\n" + "=" * 60)
print("  8. MODE")
print("=" * 60)
print(f"Most common expense category: {mode_category}")
print("\nAll categories (sorted by frequency):")
for idx in sorted_idx[:8]:  # top 8
    print(f"  {unique_categories[idx]:20s}  {category_counts[idx]} transactions")


# ===========================================================================
# 9. VARIANCE AND STANDARD DEVIATION
# ===========================================================================
# Question: "How consistent is my sleep? Do I sleep roughly the same
# every night, or is it all over the place?"
#
# VARIANCE = average of squared distances from the mean.
# Formula: var = sum((x_i - mean)^2) / (n - 1)
#
# STANDARD DEVIATION (std) = square root of variance.
# Same concept, but in the ORIGINAL units (hours, not hours^2).
#
# Why n-1? (Bessel's correction)
# When working with a SAMPLE (not the entire population), dividing by n
# underestimates the true variance. n-1 corrects this bias.
# Use n only if you have the ENTIRE population.
#
# WARNING: np.var() and np.std() use ddof=0 (population) by default!
# For sample variance/std, always pass ddof=1.

sleep_var = np.var(sleep_hours_all, ddof=1)  # sample variance
sleep_std = np.std(sleep_hours_all, ddof=1)  # sample std
sleep_mean = np.mean(sleep_hours_all)

print("\n" + "=" * 60)
print("  9. VARIANCE AND STANDARD DEVIATION")
print("=" * 60)
print(f"Sleep: mean={sleep_mean:.2f}h  var={sleep_var:.2f}h²  std={sleep_std:.2f}h")
print(
    f"  -> std={sleep_std:.2f}h means most nights are within "
    f"±{sleep_std:.2f}h of the mean"
)
print(
    f"  -> Typical range: {sleep_mean - sleep_std:.2f}h to {sleep_mean + sleep_std:.2f}h"
)

# Compare with body weight:
weight_mean = np.mean(body_weight_all)
weight_std = np.std(body_weight_all, ddof=1)
print(f"\nWeight: mean={weight_mean:.1f}kg  std={weight_std:.2f}kg")

# Compare with daily steps:
steps_mean = np.mean(daily_steps_all)
steps_std = np.std(daily_steps_all, ddof=1)
print(f"Steps:  mean={steps_mean:.0f}  std={steps_std:.0f}")
print("  -> Steps have high std relative to mean = very variable day to day")


# ===========================================================================
# 10. RANGE — max - min
# ===========================================================================
# Question: "What is the gap between my best and worst night of sleep?"
#
# Simplest measure of spread: max - min.
# Problem: very SENSITIVE to outliers (one extreme value changes everything).

sleep_range = np.max(sleep_hours_all) - np.min(sleep_hours_all)
steps_range = np.max(daily_steps_all) - np.min(daily_steps_all)

print("\n" + "=" * 60)
print("  10. RANGE")
print("=" * 60)
print(
    f"Sleep: min={np.min(sleep_hours_all):.2f}h  max={np.max(sleep_hours_all):.2f}h  "
    f"range={sleep_range:.2f}h"
)
print(
    f"Steps: min={np.min(daily_steps_all):.0f}  max={np.max(daily_steps_all):.0f}  "
    f"range={steps_range:.0f}"
)
print("  -> Range is sensitive to outliers (one extreme day changes it)")


# ===========================================================================
# 11. IQR — INTERQUARTILE RANGE
# ===========================================================================
# Question: "What is the spread of the 'middle 50%' of my data?"
# Domanda: "Quanto varia il 50% centrale dei miei dati?"
#
# IQR = Q3 - Q1, where:
#   Q1 = 25th percentile (25% of values are below this)
#   Q3 = 75th percentile (75% of values are below this)
#
# Unlike range, IQR IGNORES outliers — it only looks at the central half.
# Robust measure of spread.
#
#   |---outlier---[=====IQR=====]---outlier---|
#                Q1    median   Q3

sleep_q1 = np.percentile(sleep_hours_all, 25)
sleep_q3 = np.percentile(sleep_hours_all, 75)
sleep_iqr = sleep_q3 - sleep_q1
sleep_iqr_scipy = iqr(sleep_hours_all)  # same result, scipy shortcut

print("\n" + "=" * 60)
print("  11. IQR (INTERQUARTILE RANGE)")
print("=" * 60)
print(f"Sleep: Q1={sleep_q1:.2f}h  Q3={sleep_q3:.2f}h  IQR={sleep_iqr:.2f}h")
print(f"  -> The middle 50% of nights are between {sleep_q1:.2f}h and {sleep_q3:.2f}h")

steps_q1 = np.percentile(daily_steps_all, 25)
steps_q3 = np.percentile(daily_steps_all, 75)
steps_iqr = iqr(daily_steps_all)
print(f"Steps: Q1={steps_q1:.0f}  Q3={steps_q3:.0f}  IQR={steps_iqr:.0f}")

# Compare range vs IQR:
print(f"\nSleep — range={sleep_range:.2f}h vs IQR={sleep_iqr:.2f}h")
print(f"Steps — range={steps_range:.0f} vs IQR={steps_iqr:.0f}")
print("  -> IQR is much smaller than range = outliers inflate the range")
