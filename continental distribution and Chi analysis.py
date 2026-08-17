"""
UNGA Climate-Related Discourse:
Continental Distribution Across Diachronic Zones

This script performs two analyses:

1. Visualises the percentage distribution of climate-related sentences
   across continents and diachronic zones using:
   - Stacked bar chart
   - Clustered bar chart

2. Performs a Pearson chi-square test of independence to examine the
   association between continent and diachronic zone, including:
   - Expected frequencies
   - Cramér's V
   - Standardised Pearson residuals
   - Adjusted standardised residuals
   - Cell contributions to chi-square
   - Expected-frequency assumption checks
   - Identification of significantly over- and under-represented cells
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# ============================================================
# 1. FILE PATHS
# ============================================================

file_path = "/Users/user/Desktop/Book2.xlsx"
output_folder = "/Users/user/Desktop"

chi_square_output_file = (
    "/Users/user/Desktop/"
    "Chi_Square_Continent_by_Diachronic_Zone.xlsx"
)


# ============================================================
# 2. READ DATA
# ============================================================

df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())


# ============================================================
# 3. CLEAN VARIABLES
# ============================================================

# Clean continent names
df["Continent"] = (
    df["Continent"]
    .astype(str)
    .str.strip()
)


# Standardise diachronic-zone labels
#
# Converts:
# 1985 – 1991
# 1985-1991
# 1985 - 1991
#
# into:
# 1985-1991

df["Year"] = (
    df["Year"]
    .astype(str)
    .str.strip()
    .str.replace(
        r"\s*[–—-]\s*",
        "-",
        regex=True
    )
)


# Clean percentage values
df["Percentage"] = (
    df["Percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)

df["Percentage"] = pd.to_numeric(
    df["Percentage"],
    errors="coerce"
)


# Convert decimal percentages to percentage points if needed
# e.g., 0.25 becomes 25
if df["Percentage"].max() <= 1:
    df["Percentage"] = df["Percentage"] * 100


# Convert sentence counts to numeric
df["Sentence count"] = pd.to_numeric(
    df["Sentence count"],
    errors="coerce"
)


# ============================================================
# 4. DEFINE DIACHRONIC AND CONTINENT ORDERS
# ============================================================

zone_order = [
    "1985-1991",
    "1992-1997",
    "1998-2009",
    "2010-2015",
    "2016-2025"
]

continent_order = [
    "Africa",
    "Asia",
    "Europe",
    "Oceania",
    "Americas"
]


# Publication-style zone labels using en dashes
zone_labels = [
    "1985–1991",
    "1992–1997",
    "1998–2009",
    "2010–2015",
    "2016–2025"
]


# ============================================================
# PART I: VISUALISATION
# ============================================================


# ============================================================
# 5. CREATE PERCENTAGE PIVOT TABLE
# ============================================================

plot_data = df.pivot_table(
    index="Year",
    columns="Continent",
    values="Percentage",
    aggfunc="sum"
)

plot_data = plot_data.reindex(
    index=zone_order,
    columns=continent_order
)


# ============================================================
# 6. CHECK PERCENTAGE DATA
# ============================================================

print("\nPercentages used for plotting:")
print(plot_data.round(2))

print("\nTotal percentage for each zone:")
print(plot_data.sum(axis=1).round(2))


# ============================================================
# 7. STACKED BAR CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6.5)
)

bottom = np.zeros(len(plot_data))


for continent in continent_order:

    values = plot_data[continent].values

    ax.bar(
        zone_labels,
        values,
        bottom=bottom,
        width=0.68,
        label=continent
    )

    bottom += values


# ------------------------------------------------------------
# Format stacked bar chart
# ------------------------------------------------------------

ax.set_xlabel(
    "Diachronic zone",
    fontsize=12
)

ax.set_ylabel(
    "Percentage of climate-related sentences (%)",
    fontsize=12
)

ax.set_title(
    "Continental Distribution of Climate-Related Sentences Across Diachronic Zones",
    fontsize=14,
    pad=15
)

ax.set_ylim(0, 100)

ax.set_yticks(
    np.arange(0, 101, 20)
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.legend(
    title="Continent",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light horizontal gridlines
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)

plt.tight_layout()


# ============================================================
# 8. SAVE STACKED BAR CHART
# ============================================================

stacked_png = os.path.join(
    output_folder,
    "Continental_Percentages_Stacked_Bar_Chart.png"
)

stacked_pdf = os.path.join(
    output_folder,
    "Continental_Percentages_Stacked_Bar_Chart.pdf"
)

plt.savefig(
    stacked_png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    stacked_pdf,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. CLUSTERED BAR CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6.5)
)


# Positions of diachronic zones
x = np.arange(len(zone_order))

# Width of each bar
bar_width = 0.15

# Centre the five bars around each zone
offsets = (
    np.arange(len(continent_order))
    - (len(continent_order) - 1) / 2
) * bar_width


for i, continent in enumerate(continent_order):

    values = plot_data[continent].values

    ax.bar(
        x + offsets[i],
        values,
        width=bar_width,
        label=continent
    )


# ------------------------------------------------------------
# Format clustered bar chart
# ------------------------------------------------------------

ax.set_xticks(x)

ax.set_xticklabels(
    zone_labels,
    fontsize=10
)

ax.set_xlabel(
    "Diachronic zone",
    fontsize=12
)

ax.set_ylabel(
    "Percentage of climate-related sentences (%)",
    fontsize=12
)

ax.set_title(
    "Continental Distribution of Climate-Related Sentences Across Diachronic Zones",
    fontsize=14,
    pad=15
)

ax.tick_params(
    axis="y",
    labelsize=10
)

ax.legend(
    title="Continent",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light horizontal gridlines
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)

# Add a little space above the highest bar
max_value = plot_data.max().max()

ax.set_ylim(
    0,
    max_value * 1.12
)

plt.tight_layout()


# ============================================================
# 10. SAVE CLUSTERED BAR CHART
# ============================================================

clustered_png = os.path.join(
    output_folder,
    "Continental_Percentages_Clustered_Bar_Chart.png"
)

clustered_pdf = os.path.join(
    output_folder,
    "Continental_Percentages_Clustered_Bar_Chart.pdf"
)

plt.savefig(
    clustered_png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    clustered_pdf,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. VISUALISATION COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("CLEAN CHARTS CREATED SUCCESSFULLY")
print("=" * 70)

print("\nStacked bar chart:")
print(stacked_png)

print("\nClustered bar chart:")
print(clustered_png)


# ============================================================
# PART II: CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================


# ============================================================
# 12. CREATE OBSERVED CONTINGENCY TABLE
# ============================================================

# Rows = diachronic zones
# Columns = continents
# Values = observed sentence counts

observed = df.pivot_table(
    index="Year",
    columns="Continent",
    values="Sentence count",
    aggfunc="sum",
    fill_value=0
)


# Put rows and columns in the required order
observed = observed.reindex(
    index=zone_order,
    columns=continent_order,
    fill_value=0
)


print("\n" + "=" * 70)
print("OBSERVED CONTINGENCY TABLE")
print("=" * 70)

print(observed)


# ============================================================
# 13. CHECK SENTENCE TOTALS
# ============================================================

print("\nSentence totals by zone:")
print(observed.sum(axis=1))

print("\nSentence totals by continent:")
print(observed.sum(axis=0))

grand_total = observed.to_numpy().sum()

print(f"\nTotal number of sentences: {grand_total:,}")


# ============================================================
# 14. PEARSON CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================

chi2, p_value, dof, expected_array = chi2_contingency(
    observed,
    correction=False
)


# Convert expected values into a DataFrame
expected = pd.DataFrame(
    expected_array,
    index=observed.index,
    columns=observed.columns
)


# ============================================================
# 15. CRAMÉR'S V
# ============================================================

n = observed.to_numpy().sum()

r, c = observed.shape

cramers_v = np.sqrt(
    chi2
    /
    (
        n * min(r - 1, c - 1)
    )
)


# ============================================================
# 16. STANDARDISED PEARSON RESIDUALS
# ============================================================

# Positive residual:
# More sentences than expected
#
# Negative residual:
# Fewer sentences than expected

standardised_residuals = (
    observed - expected
) / np.sqrt(expected)


# ============================================================
# 17. ADJUSTED STANDARDISED RESIDUALS
# ============================================================

# Adjusted standardised residuals account for the row and
# column marginal distributions and help identify which cells
# contribute to the overall chi-square association.

observed_array = observed.to_numpy()

row_totals = observed_array.sum(
    axis=1,
    keepdims=True
)

column_totals = observed_array.sum(
    axis=0,
    keepdims=True
)

row_proportions = row_totals / n
column_proportions = column_totals / n


adjusted_residual_array = (
    observed_array - expected_array
) / np.sqrt(
    expected_array
    *
    (1 - row_proportions)
    *
    (1 - column_proportions)
)


adjusted_residuals = pd.DataFrame(
    adjusted_residual_array,
    index=observed.index,
    columns=observed.columns
)


# ============================================================
# 18. CELL CONTRIBUTIONS TO CHI-SQUARE
# ============================================================

chi_square_contributions = (
    (observed - expected) ** 2
) / expected


# ============================================================
# 19. CHECK EXPECTED-FREQUENCY ASSUMPTION
# ============================================================

number_cells = expected.size

cells_below_5 = (
    expected < 5
).sum().sum()

cells_below_1 = (
    expected < 1
).sum().sum()

percentage_below_5 = (
    cells_below_5 / number_cells
) * 100

minimum_expected = expected.min().min()


# ============================================================
# 20. CLASSIFY ADJUSTED STANDARDISED RESIDUALS
# ============================================================

# Conventional two-sided 5% threshold:
# |residual| > 1.96

residual_flags = adjusted_residuals.copy()


def classify_residual(value):
    """
    Classify an adjusted standardised residual.

    Parameters
    ----------
    value : float
        Adjusted standardised residual.

    Returns
    -------
    str
        Over-represented, under-represented, or not significant.
    """

    if value > 1.96:
        return "Over-represented"

    elif value < -1.96:
        return "Under-represented"

    else:
        return "Not significant"


residual_flags = residual_flags.map(
    classify_residual
)


# ============================================================
# 21. CREATE LONG-FORM RESIDUAL TABLE
# ============================================================

residual_long = []


for zone in zone_order:

    for continent in continent_order:

        residual_long.append(
            {
                "Zone": zone,
                "Continent": continent,
                "Observed": observed.loc[
                    zone,
                    continent
                ],
                "Expected": expected.loc[
                    zone,
                    continent
                ],
                "Standardised_Residual":
                    standardised_residuals.loc[
                        zone,
                        continent
                    ],
                "Adjusted_Standardised_Residual":
                    adjusted_residuals.loc[
                        zone,
                        continent
                    ],
                "Chi_Square_Contribution":
                    chi_square_contributions.loc[
                        zone,
                        continent
                    ],
                "Interpretation":
                    residual_flags.loc[
                        zone,
                        continent
                    ]
            }
        )


residual_long = pd.DataFrame(
    residual_long
)


# Sort by absolute adjusted residual
residual_long["Absolute_Adjusted_Residual"] = (
    residual_long[
        "Adjusted_Standardised_Residual"
    ].abs()
)

residual_long = residual_long.sort_values(
    "Absolute_Adjusted_Residual",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 22. CREATE OVERALL RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    {
        "Statistic": [
            "Chi-square",
            "Degrees of freedom",
            "p-value",
            "Cramer's V",
            "N",
            "Minimum expected count",
            "Cells with expected count < 5",
            "Percentage of cells < 5",
            "Cells with expected count < 1"
        ],

        "Value": [
            chi2,
            dof,
            p_value,
            cramers_v,
            n,
            minimum_expected,
            cells_below_5,
            percentage_below_5,
            cells_below_1
        ]
    }
)


# ============================================================
# 23. PRINT OVERALL CHI-SQUARE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 70)

print(
    f"Chi-square: χ²({dof}) = {chi2:.2f}"
)

if p_value < .001:

    print("p-value: p < .001")

else:

    print(
        f"p-value: p = {p_value:.4f}"
    )

print(
    f"Cramér's V = {cramers_v:.3f}"
)

print(
    f"N = {n:,}"
)


# ============================================================
# 24. PRINT EXPECTED-FREQUENCY ASSUMPTION CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED FREQUENCY CHECK")
print("=" * 70)

print(
    f"Minimum expected frequency: "
    f"{minimum_expected:.2f}"
)

print(
    f"Cells with expected frequency < 5: "
    f"{cells_below_5} of {number_cells} "
    f"({percentage_below_5:.1f}%)"
)

print(
    f"Cells with expected frequency < 1: "
    f"{cells_below_1}"
)


# ============================================================
# 25. PRINT EXPECTED COUNTS
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED COUNTS")
print("=" * 70)

print(
    expected.round(2)
)


# ============================================================
# 26. PRINT ADJUSTED STANDARDISED RESIDUALS
# ============================================================

print("\n" + "=" * 70)
print("ADJUSTED STANDARDISED RESIDUALS")
print("=" * 70)

print(
    adjusted_residuals.round(2)
)


# ============================================================
# 27. IDENTIFY AND PRINT SIGNIFICANT CELLS
# ============================================================

significant_cells = residual_long[
    residual_long[
        "Absolute_Adjusted_Residual"
    ] > 1.96
]


print("\n" + "=" * 70)
print("CELLS WITH |ADJUSTED RESIDUAL| > 1.96")
print("=" * 70)

print(
    significant_cells[
        [
            "Zone",
            "Continent",
            "Observed",
            "Expected",
            "Adjusted_Standardised_Residual",
            "Interpretation"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 28. SAVE CHI-SQUARE RESULTS TO EXCEL
# ============================================================

with pd.ExcelWriter(
    chi_square_output_file,
    engine="openpyxl"
) as writer:

    # Overall test
    results.to_excel(
        writer,
        sheet_name="Chi_Square_Results",
        index=False
    )

    # Observed counts
    observed.to_excel(
        writer,
        sheet_name="Observed_Counts"
    )

    # Expected counts
    expected.to_excel(
        writer,
        sheet_name="Expected_Counts"
    )

    # Standardised residuals
    standardised_residuals.to_excel(
        writer,
        sheet_name="Standardised_Residuals"
    )

    # Adjusted residuals
    adjusted_residuals.to_excel(
        writer,
        sheet_name="Adjusted_Residuals"
    )

    # Cell contributions
    chi_square_contributions.to_excel(
        writer,
        sheet_name="ChiSq_Contributions"
    )

    # Long-form cell analysis
    residual_long.to_excel(
        writer,
        sheet_name="Cell_Analysis",
        index=False
    )

    # Significant cells only
    significant_cells.to_excel(
        writer,
        sheet_name="Significant_Cells",
        index=False
    )


# ============================================================
# 29. ANALYSIS COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"\nResults saved to:\n{chi_square_output_file}"
)