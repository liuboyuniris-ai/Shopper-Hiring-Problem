import pandas as pd
import numpy as np
import scipy.stats as stats
df=pd.read_csv('application.csv')
control_unique_count = df[df['group'] == 'control']['applicant_id'].nunique()
treatment_unique_count = df[df['group'] == 'treatment']['applicant_id'].nunique()

# output result
print(f"Control Group unique ID count: {control_unique_count}")
print(f"Treatment Group unique ID count: {treatment_unique_count}")

control_converted = df[(df['group'] == 'control') & (df['event'] == 'first_batch_completed_date')]['applicant_id'].nunique()
treatment_converted = df[(df['group'] == 'treatment') & (df['event'] == 'first_batch_completed_date')]['applicant_id'].nunique()
print(f'control group conversion count:',control_converted)
print(f'treatment group conversion count:',treatment_converted)

# Load data
df = pd.read_csv("application.csv")

# Calculate unique applicant counts
control_unique_count = df[df['group'] == 'control']['applicant_id'].nunique()
treatment_unique_count = df[df['group'] == 'treatment']['applicant_id'].nunique()

# Calculate conversions
control_converted = df[(df['group'] == 'control') &
                       (df['event'] == 'first_batch_completed_date')]['applicant_id'].nunique()
treatment_converted = df[(df['group'] == 'treatment') &
                         (df['event'] == 'first_batch_completed_date')]['applicant_id'].nunique()

# Conversion rates
control_rate = control_converted / control_unique_count
treatment_rate = treatment_converted / treatment_unique_count
diff = treatment_rate - control_rate

# Pooled proportion
p_pool = (control_converted + treatment_converted) / (control_unique_count + treatment_unique_count)

# Standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1/control_unique_count + 1/treatment_unique_count))

# Z-score
z_score = (treatment_rate - control_rate) / se

# Two-tailed p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# 95% Confidence Interval
se_diff = np.sqrt((control_rate*(1-control_rate)/control_unique_count) +
                  (treatment_rate*(1-treatment_rate)/treatment_unique_count))
ci_low = diff - 1.96 * se_diff
ci_high = diff + 1.96 * se_diff

print(f"Control Conversion Rate: {control_rate:.4f}")
print(f"Treatment Conversion Rate: {treatment_rate:.4f}")
print(f"Difference: {diff:.4f}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"95% CI for Difference: [{ci_low:.4f}, {ci_high:.4f}]")

# Keep only application and completion events
df_filtered = df[df["event"].isin(["application_date", "first_batch_completed_date"])].copy()

# Convert to wide format: each applicant_id gets both dates as columns
df_pivot = df_filtered.pivot_table(index=["applicant_id", "group"],
                                   columns="event",
                                   values="event_date",  # assuming you have a 'date' column for event timestamp
                                   aggfunc="first").reset_index()

# Convert to datetime
df_pivot["application_date"] = pd.to_datetime(df_pivot["application_date"], errors="coerce")
df_pivot["first_batch_completed_date"] = pd.to_datetime(df_pivot["first_batch_completed_date"], errors="coerce")

# Compute days difference
df_pivot["days_to_completion"] = (df_pivot["first_batch_completed_date"] -
                                  df_pivot["application_date"]).dt.days

# Calculate average days by group
avg_days = df_pivot.groupby("group")["days_to_completion"].mean()

print("Average days from application to first batch completion:")
print(avg_days)

# Q2 Whether it is cost effective

import pandas as pd

# Load data
df = pd.read_csv("application.csv")

# Cost per background check
background_check_cost = 30

# Calculate applicant counts per group
applicants = df.groupby("group")["applicant_id"].nunique()

# Calculate completions (first batch)
completions = df[df["event"] == "first_batch_completed_date"].groupby("group")["applicant_id"].nunique()

# Merge into a summary DataFrame
summary = pd.DataFrame({
    "applicants": applicants,
    "completions": completions
})

# Calculate costs
summary["total_cost"] = summary["applicants"] * background_check_cost

# Cost per successful shopper
summary["cost_per_success"] = summary["total_cost"] / summary["completions"]

# Conversion rate for reference
summary["conversion_rate"] = summary["completions"] / summary["applicants"]

# (Optional) Incremental cost-effectiveness ratio (ICER)
# Difference in cost divided by difference in completions
icer = ((summary.loc["treatment", "total_cost"] - summary.loc["control", "total_cost"]) /
        (summary.loc["treatment", "completions"] - summary.loc["control", "completions"]))

print("Cost-effectiveness summary by group:")
print(summary)
print(f"\nIncremental Cost-Effectiveness Ratio (ICER): ${icer:.2f} per additional successful shopper")

