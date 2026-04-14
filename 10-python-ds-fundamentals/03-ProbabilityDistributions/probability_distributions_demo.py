"""
Probability Distributions on real personal datasets.
Datasets: health (weight/sleep), wallet (expenses), workout (warmup sets).
Focus: understanding distributions, fitting real data, using scipy.stats.
"""

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, expon, norm, poisson

# Add parent directory to path so we can import the shared data_loader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# pylint: disable=import-error,wrong-import-position
from data_loader import load_csv  # noqa: E402
# pylint: enable=import-error,wrong-import-position


# ===========================================================================
# 0. LOADING DATA
# ===========================================================================

wallet_data = load_csv("wallet.csv")
health_data = load_csv("health.csv")
workout_data = load_csv("workoutExport.csv")

# Body weight — continuous data, good candidate for Normal distribution
# body_weight_all → [77.9, 77.7, 77.5, ...]  (kg, one per measurement)
body_weight_all = np.array([
    float(row["weight_kg"])
    for row in health_data
    if row["weight_kg"]
])

# Sleep hours — continuous data
# sleep_hours_all → [7.24, 6.54, ...]  (hours per night)
sleep_hours_all = np.array([
    float(row["sleep_total_hr"])
    for row in health_data
    if row["sleep_total_hr"]
])

# Expense amounts (absolute value, only actual expenses)
# expenses_all → [7.5, 90.8, 10.0, 20.0, ...]  (EUR per transaction)
expenses_all = np.array([
    abs(float(row["amount"]))
    for row in wallet_data
    if row["amount"] and float(row["amount"]) < 0
])

# Expense dates (only actual expenses, not income)
expense_dates = sorted([
    row["date"]
    for row in wallet_data
    if row["amount"] and float(row["amount"]) < 0
])

# Warmup flags from workout data
# warmup_flags → [False, False, True, False, ...]  (one per set)
warmup_flags = np.array([
    row["isWarmup"] == "true"
    for row in workout_data
    if row["isWarmup"]
])

print(f"Body weight: {len(body_weight_all)} measurements")
print(f"Sleep hours: {len(sleep_hours_all)} nights")
print(f"Expense records: {len(expense_dates)}")
print(f"Workout sets: {len(warmup_flags)} ({warmup_flags.sum()} warmup)")


# ===========================================================================
# ===========================================================================
#
#   WHAT IS A PROBABILITY DISTRIBUTION?
#
# ===========================================================================
# ===========================================================================
# A probability distribution describes HOW the values of a random variable
# are spread out. It tells you: "what values are likely, and how likely?"
#
# Two types:
#   - Discrete: countable values (number of transactions, number of reps)
#   - Continuous: any value in a range (weight, sleep hours, speed)
#
# Key functions (scipy.stats provides all of these):
#   .pdf(x)  → probability density at x       (continuous)
#   .pmf(x)  → probability mass at x          (discrete)
#   .cdf(x)  → P(X ≤ x) — cumulative probability
#   .ppf(p)  → inverse of cdf: "what value has cumulative probability p?"
#   .mean()  → expected value
#   .std()   → standard deviation


# ===========================================================================
# 1. NORMAL (GAUSSIAN) DISTRIBUTION — scipy.stats.norm
# ===========================================================================
# Question: "Does my body weight follow a bell curve?"
#
# The most important distribution in statistics.
# Describes phenomena where values cluster symmetrically around a mean.
# Parameters: μ (mean) and σ (standard deviation)
#
# The 68-95-99.7 rule:
#   68% of values fall within [μ-σ,  μ+σ]    (1 std from mean)
#   95% of values fall within [μ-2σ, μ+2σ]   (2 std from mean)
#   99.7% of values fall within [μ-3σ, μ+3σ]  (3 std from mean)

# Fit a normal distribution to real body weight data
weight_mean = np.mean(body_weight_all)
weight_std = np.std(body_weight_all, ddof=1)

print("\n" + "=" * 60)
print("  1. NORMAL (GAUSSIAN) DISTRIBUTION")
print("=" * 60)
print(f"Body weight: μ={weight_mean:.1f} kg, σ={weight_std:.2f} kg")

# Verify the 68-95-99.7 rule on real data:
within_1_std = np.sum(
    (body_weight_all >= weight_mean - weight_std)
    & (body_weight_all <= weight_mean + weight_std)
)
within_2_std = np.sum(
    (body_weight_all >= weight_mean - 2 * weight_std)
    & (body_weight_all <= weight_mean + 2 * weight_std)
)
within_3_std = np.sum(
    (body_weight_all >= weight_mean - 3 * weight_std)
    & (body_weight_all <= weight_mean + 3 * weight_std)
)
n = len(body_weight_all)

print(f"\n68-95-99.7 rule on body weight ({n} measurements):")
print(f"  Within 1σ ({weight_mean - weight_std:.1f}–{weight_mean + weight_std:.1f} kg): "
      f"{within_1_std / n:.1%}  (expected: 68%)")
print(f"  Within 2σ ({weight_mean - 2 * weight_std:.1f}–{weight_mean + 2 * weight_std:.1f} kg): "
      f"{within_2_std / n:.1%}  (expected: 95%)")
print(f"  Within 3σ ({weight_mean - 3 * weight_std:.1f}–{weight_mean + 3 * weight_std:.1f} kg): "
      f"{within_3_std / n:.1%}  (expected: 99.7%)")

# Using scipy.stats.norm for probability calculations:
# "What is the probability that my weight is below 80 kg?"
weight_dist = norm(loc=weight_mean, scale=weight_std)
# .cdf(x) = P(X ≤ x) — cumulative distribution function
prob_below_80 = weight_dist.cdf(80)
# .pdf(x) = probability density at x (height of the bell curve at that point)
density_at_mean = weight_dist.pdf(weight_mean)
# .ppf(p) = "what weight corresponds to the 95th percentile?"
weight_95th = weight_dist.ppf(0.95)

print(f"\nUsing scipy.stats.norm (fitted to real data):")
print(f"  P(weight < 80 kg) = {prob_below_80:.2%}")
print(f"  Density at mean:    {density_at_mean:.4f}")
print(f"  95th percentile:    {weight_95th:.1f} kg")

# Same analysis on sleep data
sleep_mean = np.mean(sleep_hours_all)
sleep_std = np.std(sleep_hours_all, ddof=1)
sleep_dist = norm(loc=sleep_mean, scale=sleep_std)

print(f"\nSleep: μ={sleep_mean:.2f}h, σ={sleep_std:.2f}h")
print(f"  P(sleep < 4h) = {sleep_dist.cdf(4):.2%}")
print(f"  P(sleep > 7h) = {1 - sleep_dist.cdf(7):.2%}")

# --- PLOT: histogram of body weight + fitted normal curve ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: body weight
ax = axes[0]
ax.hist(body_weight_all, bins=25, density=True, alpha=0.7, color="steelblue", label="Real data")
x_weight = np.linspace(weight_mean - 4 * weight_std, weight_mean + 4 * weight_std, 200)
ax.plot(x_weight, weight_dist.pdf(x_weight), "r-", linewidth=2, label="Normal fit")
ax.axvline(weight_mean, color="orange", linestyle="--", label=f"μ={weight_mean:.1f}")
ax.set_title("Body Weight — Normal Distribution")
ax.set_xlabel("Weight (kg)")
ax.set_ylabel("Density")
ax.legend()

# Right: sleep hours
ax = axes[1]
ax.hist(sleep_hours_all, bins=30, density=True, alpha=0.7, color="steelblue", label="Real data")
x_sleep = np.linspace(0, 12, 200)
ax.plot(x_sleep, sleep_dist.pdf(x_sleep), "r-", linewidth=2, label="Normal fit")
ax.axvline(sleep_mean, color="orange", linestyle="--", label=f"μ={sleep_mean:.2f}")
ax.set_title("Sleep Hours — Normal Distribution")
ax.set_xlabel("Hours")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_1_normal.png", dpi=150)
plt.show()


# ===========================================================================
# 2. EXPONENTIAL DISTRIBUTION — scipy.stats.expon
# ===========================================================================
# Question: "How many days between one expense and the next?"
#
# Describes the time/distance between events in a Poisson process.
# Examples: time between arrivals, server requests, purchases.
# Parameter: λ (rate). Mean = 1/λ.
#
# IMPORTANT: scipy and numpy use scale = 1/λ, NOT λ directly.
# Python stdlib random.expovariate() uses λ. This is a common source of bugs.

# Calculate days between expenses (real data)
expense_datetimes = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(set(expense_dates))]
# days_between_expenses → [1, 1, 1, 2, 1, 3, ...]  (days between consecutive expense days)
days_between_expenses = np.array([
    (expense_datetimes[i + 1] - expense_datetimes[i]).days
    for i in range(len(expense_datetimes) - 1)
])

# Fit exponential: the mean gap tells us the scale parameter
mean_gap = np.mean(days_between_expenses)
# λ (rate) = 1/mean_gap, scale = mean_gap = 1/λ
expense_rate = 1 / mean_gap

print("\n" + "=" * 60)
print("  2. EXPONENTIAL DISTRIBUTION")
print("=" * 60)
print(f"Days between expenses: mean={mean_gap:.2f} days (λ={expense_rate:.2f}/day)")

# Using scipy.stats.expon:
exp_dist = expon(scale=mean_gap)  # scale = 1/λ = mean
print(f"\nUsing scipy.stats.expon:")
# "What is the probability of waiting more than 3 days between expenses?"
prob_more_3 = 1 - exp_dist.cdf(3)
print(f"  P(gap > 3 days) = {prob_more_3:.2%}")
# "What is the probability of an expense within 1 day?"
prob_within_1 = exp_dist.cdf(1)
print(f"  P(gap ≤ 1 day)  = {prob_within_1:.2%}")

# Verify with real data:
real_more_3 = np.sum(days_between_expenses > 3) / len(days_between_expenses)
real_within_1 = np.sum(days_between_expenses <= 1) / len(days_between_expenses)
print(f"\nReal data comparison:")
print(f"  Real P(gap > 3 days) = {real_more_3:.2%}")
print(f"  Real P(gap ≤ 1 day)  = {real_within_1:.2%}")

# --- PLOT: histogram of gaps + fitted exponential curve ---
fig, ax = plt.subplots(figsize=(8, 5))
max_gap = int(days_between_expenses.max())
ax.hist(days_between_expenses, bins=range(0, max_gap + 2), density=True,
        alpha=0.7, color="steelblue", label="Real data", edgecolor="white")
x_exp = np.linspace(0, max_gap, 200)
ax.plot(x_exp, exp_dist.pdf(x_exp), "r-", linewidth=2, label="Exponential fit")
ax.set_title("Days Between Expenses — Exponential Distribution")
ax.set_xlabel("Days")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_2_exponential.png", dpi=150)
plt.show()


# ===========================================================================
# 3. POISSON DISTRIBUTION — scipy.stats.poisson
# ===========================================================================
# Question: "How many transactions do I make per day?"
#
# Counts the number of events in a fixed interval.
# Discrete distribution: values are 0, 1, 2, 3, ...
# Parameter: λ (average number of events per interval). Mean = λ, Var = λ.
#
# Key property: if events follow a Poisson process, the TIME between events
# follows an Exponential distribution (section 2 above).

# Count transactions per day (real data)
tx_per_day_counter = Counter(expense_dates)
# tx_per_day_all → [1, 2, 3, 5, 1, 4, ...]  (transactions per day)
tx_per_day_all = np.array(list(tx_per_day_counter.values()))

lambda_tx = np.mean(tx_per_day_all)

print("\n" + "=" * 60)
print("  3. POISSON DISTRIBUTION")
print("=" * 60)
print(f"Transactions per day: λ={lambda_tx:.2f}")

# Using scipy.stats.poisson:
pois_dist = poisson(mu=lambda_tx)
# .pmf(k) = P(X = k) — probability of exactly k transactions
# (pmf = probability mass function, used for DISCRETE distributions
#  instead of pdf which is for continuous)
print(f"\nUsing scipy.stats.poisson:")
for k in range(7):
    # Model prediction vs actual frequency
    predicted = pois_dist.pmf(k)
    actual = np.sum(tx_per_day_all == k) / len(tx_per_day_all)
    print(f"  P(X={k}) = {predicted:.3f} (model)  vs  {actual:.3f} (real)")

# "What is the probability of 6 or more transactions in a day?"
prob_6_plus = 1 - pois_dist.cdf(5)  # P(X > 5) = 1 - P(X ≤ 5)
real_6_plus = np.sum(tx_per_day_all >= 6) / len(tx_per_day_all)
print(f"\n  P(X ≥ 6) = {prob_6_plus:.3f} (model)  vs  {real_6_plus:.3f} (real)")

# --- PLOT: Poisson model vs real data (bar chart) ---
fig, ax = plt.subplots(figsize=(8, 5))
max_k = int(tx_per_day_all.max())
k_values = np.arange(0, max_k + 1)
real_freqs = np.array([np.sum(tx_per_day_all == k) / len(tx_per_day_all) for k in k_values])
model_probs = pois_dist.pmf(k_values)

bar_width = 0.35
ax.bar(k_values - bar_width / 2, real_freqs, bar_width, alpha=0.7,
       color="steelblue", label="Real data")
ax.bar(k_values + bar_width / 2, model_probs, bar_width, alpha=0.7,
       color="tomato", label="Poisson model")
ax.set_title(f"Transactions per Day — Poisson(λ={lambda_tx:.2f})")
ax.set_xlabel("Transactions")
ax.set_ylabel("Probability")
ax.set_xticks(k_values)
ax.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_3_poisson.png", dpi=150)
plt.show()


# ===========================================================================
# 4. BINOMIAL DISTRIBUTION — scipy.stats.binom
# ===========================================================================
# Question: "In a workout of 20 sets, how many will be warmup?"
#
# Counts the number of successes in n independent trials.
# Discrete distribution: values are 0, 1, 2, ..., n.
# Parameters: n (number of trials), p (probability of success per trial).
#
# Real example: each set in a workout is either warmup (True) or working (False).

# Real warmup probability from data
p_warmup = warmup_flags.sum() / len(warmup_flags)
n_sets = 20  # typical workout session

print("\n" + "=" * 60)
print("  4. BINOMIAL DISTRIBUTION")
print("=" * 60)
print(f"Warmup probability: p={p_warmup:.3f} ({warmup_flags.sum()}/{len(warmup_flags)} sets)")
print(f"In a session of {n_sets} sets:")

# Using scipy.stats.binom:
binom_dist = binom(n=n_sets, p=p_warmup)
print(f"\nUsing scipy.stats.binom (n={n_sets}, p={p_warmup:.3f}):")
print(f"  Expected warmup sets: {binom_dist.mean():.1f}")
print(f"  Std: {binom_dist.std():.1f}")

# Probabilities of specific outcomes
for k in range(6):
    print(f"  P(exactly {k} warmup) = {binom_dist.pmf(k):.3f}")

# "What is the probability of 5 or more warmup sets?"
prob_5_plus = 1 - binom_dist.cdf(4)
print(f"  P(≥ 5 warmup) = {prob_5_plus:.3f}")

# --- PLOT: Binomial distribution (bar chart) ---
fig, ax = plt.subplots(figsize=(8, 5))
k_binom = np.arange(0, n_sets + 1)
ax.bar(k_binom, binom_dist.pmf(k_binom), alpha=0.7, color="steelblue", edgecolor="white")
ax.axvline(binom_dist.mean(), color="orange", linestyle="--",
           label=f"Expected: {binom_dist.mean():.1f} warmup sets")
ax.set_title(f"Warmup Sets per Session — Binomial(n={n_sets}, p={p_warmup:.3f})")
ax.set_xlabel("Number of warmup sets")
ax.set_ylabel("Probability")
ax.set_xlim(-0.5, 10)  # zoom on relevant range
ax.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_4_binomial.png", dpi=150)
plt.show()


# ===========================================================================
# 5. OTHER DISTRIBUTIONS — QUICK REFERENCE
# ===========================================================================
# | Distribution | Use case                              | scipy          |
# |-------------|---------------------------------------|----------------|
# | Uniform     | Equal probability sampling            | uniform        |
# | Binomial    | n trials, count successes             | binom          |
# | Poisson     | Events in an interval                 | poisson        |
# | Exponential | Time between events                   | expon          |
# | Normal      | Natural phenomena, bell curve         | norm           |
# | Beta        | Probabilities, proportions [0,1]      | beta           |
# | Chi-squared | Goodness of fit tests                 | chi2           |
# | t-Student   | Inference on small samples            | t              |

# Quick demo: generate samples from each distribution
rng = np.random.default_rng(seed=42)

print("\n" + "=" * 60)
print("  5. OTHER DISTRIBUTIONS — SAMPLES")
print("=" * 60)

uniform_sample = rng.uniform(low=0, high=1, size=5)
print(f"Uniform(0,1):     {uniform_sample.round(3)}")

normal_sample = rng.normal(loc=0, scale=1, size=5)
print(f"Normal(0,1):      {normal_sample.round(3)}")

poisson_sample = rng.poisson(lam=3, size=5)
print(f"Poisson(λ=3):     {poisson_sample}")

exponential_sample = rng.exponential(scale=2, size=5)
print(f"Exponential(λ=0.5): {exponential_sample.round(3)}")

binomial_sample = rng.binomial(n=10, p=0.3, size=5)
print(f"Binomial(n=10,p=0.3): {binomial_sample}")


# ===========================================================================
# 6. LOG-NORMAL — EXPENSE AMOUNTS
# ===========================================================================
# Question: "Why do my expenses look nothing like a bell curve?"
#
# Most expenses are small (coffee, lunch), but a few are huge (rent, transfers).
# The raw data is heavily skewed → NOT normal.
# But if you take the LOG of each expense, it becomes roughly normal.
# This is called a LOG-NORMAL distribution.

from scipy.stats import lognorm  # pylint: disable=ungrouped-imports

log_expenses = np.log(expenses_all[expenses_all > 0])
log_mean = log_expenses.mean()
log_std = log_expenses.std()

# Fit a lognormal distribution
shape, loc, scale = lognorm.fit(expenses_all[expenses_all > 0], floc=0)

print("\n" + "=" * 60)
print("  6. LOG-NORMAL — EXPENSE AMOUNTS")
print("=" * 60)
print(f"Raw expenses: mean={np.mean(expenses_all):.1f}, median={np.median(expenses_all):.1f}")
print(f"  -> mean >> median = heavily skewed (NOT normal)")
print(f"Log(expenses): mean={log_mean:.2f}, std={log_std:.2f}")
print(f"  -> After log transform, much more symmetric")

# --- PLOT: raw expenses vs log(expenses) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: raw expenses (skewed)
ax = axes[0]
ax.hist(expenses_all, bins=50, density=True, alpha=0.7, color="steelblue", label="Real data")
ax.set_title("Expense Amounts — Raw (skewed)")
ax.set_xlabel("EUR")
ax.set_ylabel("Density")
ax.set_xlim(0, 300)  # zoom on bulk, outliers are off-chart
ax.legend()

# Right: log(expenses) → looks normal
ax = axes[1]
ax.hist(log_expenses, bins=30, density=True, alpha=0.7, color="steelblue", label="Real data")
x_log = np.linspace(log_expenses.min(), log_expenses.max(), 200)
log_dist = norm(loc=log_mean, scale=log_std)
ax.plot(x_log, log_dist.pdf(x_log), "r-", linewidth=2, label="Normal fit")
ax.set_title("Log(Expenses) — Approximately Normal")
ax.set_xlabel("log(EUR)")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_5_lognormal.png", dpi=150)
plt.show()


# ===========================================================================
# 7. DAILY STEPS — IS IT NORMAL?
# ===========================================================================
# Question: "Do my daily steps follow a Gaussian distribution?"
#
# Daily steps have mean ≈ median, which suggests symmetry.
# Let's visually compare the histogram with a fitted normal curve.

# Load steps (already loaded as daily_steps_all in section 0)
# Reuse from health_data
daily_steps_all = np.array([
    float(row["steps"])
    for row in health_data
    if row["steps"]
])

steps_mean = np.mean(daily_steps_all)
steps_std = np.std(daily_steps_all, ddof=1)
steps_dist = norm(loc=steps_mean, scale=steps_std)

print("\n" + "=" * 60)
print("  7. DAILY STEPS — NORMALITY CHECK")
print("=" * 60)
print(f"Steps: μ={steps_mean:.0f}, σ={steps_std:.0f}")
print(f"Mean/Median ratio: {steps_mean / np.median(daily_steps_all):.2f}x")
print(f"  -> Close to 1.0 = roughly symmetric")

# Verify 68-95-99.7 rule
within_1 = np.sum((daily_steps_all >= steps_mean - steps_std)
                   & (daily_steps_all <= steps_mean + steps_std)) / len(daily_steps_all)
within_2 = np.sum((daily_steps_all >= steps_mean - 2 * steps_std)
                   & (daily_steps_all <= steps_mean + 2 * steps_std)) / len(daily_steps_all)
print(f"  Within 1σ: {within_1:.1%} (expected 68%)")
print(f"  Within 2σ: {within_2:.1%} (expected 95%)")

# --- PLOT: steps histogram + normal fit ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(daily_steps_all, bins=40, density=True, alpha=0.7, color="steelblue", label="Real data")
x_steps = np.linspace(0, daily_steps_all.max(), 200)
ax.plot(x_steps, steps_dist.pdf(x_steps), "r-", linewidth=2, label="Normal fit")
ax.axvline(steps_mean, color="orange", linestyle="--", label=f"μ={steps_mean:.0f}")
ax.set_title("Daily Steps — Normal Distribution?")
ax.set_xlabel("Steps")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_6_steps_normal.png", dpi=150)
plt.show()


# ===========================================================================
# 8. WORKOUT SESSION GAPS — EXPONENTIAL
# ===========================================================================
# Question: "How many days between workout sessions?"
#
# Similar to expense gaps (section 2), but workout frequency is less regular.

workout_session_dates = sorted({row["Date"][:10] for row in workout_data})
workout_datetimes = [datetime.strptime(d, "%Y-%m-%d") for d in workout_session_dates]
workout_gaps = np.array([
    (workout_datetimes[i + 1] - workout_datetimes[i]).days
    for i in range(len(workout_datetimes) - 1)
])

workout_mean_gap = np.mean(workout_gaps)
workout_rate = 1 / workout_mean_gap
workout_exp_dist = expon(scale=workout_mean_gap)

print("\n" + "=" * 60)
print("  8. WORKOUT GAPS — EXPONENTIAL")
print("=" * 60)
print(f"Workout sessions: {len(workout_session_dates)}")
print(f"Days between sessions: mean={workout_mean_gap:.2f} (λ={workout_rate:.2f}/day)")
print(f"  P(gap > 3 days) = {1 - workout_exp_dist.cdf(3):.2%} (model) vs "
      f"{np.sum(workout_gaps > 3) / len(workout_gaps):.2%} (real)")
print(f"  P(gap > 7 days) = {1 - workout_exp_dist.cdf(7):.2%} (model) vs "
      f"{np.sum(workout_gaps > 7) / len(workout_gaps):.2%} (real)")

# --- PLOT: workout gaps histogram + exponential fit ---
fig, ax = plt.subplots(figsize=(8, 5))
max_gap = min(int(workout_gaps.max()), 15)  # zoom on relevant range
ax.hist(workout_gaps, bins=range(0, max_gap + 2), density=True,
        alpha=0.7, color="steelblue", label="Real data", edgecolor="white")
x_wk = np.linspace(0, max_gap, 200)
ax.plot(x_wk, workout_exp_dist.pdf(x_wk), "r-", linewidth=2, label="Exponential fit")
ax.set_title(f"Days Between Workouts — Exponential(λ={workout_rate:.2f})")
ax.set_xlabel("Days")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_7_workout_gaps.png", dpi=150)
plt.show()


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================

print("\n" + "=" * 60)
print("  SUMMARY: DISTRIBUTIONS FITTED TO REAL DATA")
print("=" * 60)
print(f"  Body weight    ~ Normal(μ={weight_mean:.1f}, σ={weight_std:.2f})")
print(f"  Daily steps    ~ Normal(μ={steps_mean:.0f}, σ={steps_std:.0f})")
print(f"  Expenses       ~ Log-Normal (log: μ={log_mean:.2f}, σ={log_std:.2f})")
print(f"  Expense gaps   ~ Exponential(λ={expense_rate:.2f})")
print(f"  Workout gaps   ~ Exponential(λ={workout_rate:.2f})")
print(f"  Tx per day     ~ Poisson(λ={lambda_tx:.2f})")
print(f"  Warmup sets    ~ Binomial(n=20, p={p_warmup:.3f})")
