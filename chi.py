import numpy as np
from scipy.stats import chi2  

observed = [25, 30, 20, 25]  # these are your actual counts
expected = [25, 25, 25, 25]  # what you expected under the null hypothesis

# Step 1: Check data validity
if len(observed) != len(expected):
    raise ValueError("Observed and expected should be of same length")

# Step 2: Calculate Chi-Square statistic manually
chi_square_stat = 0
for o, e in zip(observed, expected):
    if e == 0:
        raise ValueError("Expected frequency can't be zero.")
    chi_square_stat += ((o - e) ** 2) / e

# Step 3: Degrees of freedom
df = len(observed) - 1

# Step 4: Calculate p-value from chi-squared distribution
p_value = 1 - chi2.cdf(chi_square_stat, df)

# Step 5: Print results
print("Chi-Square Statistic:", round(chi_square_stat, 4))
print("Degrees of Freedom:", df)
print("P-Value:", round(p_value, 4))

# Optional: interpret result at alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Can't reject the null hypothesis")
