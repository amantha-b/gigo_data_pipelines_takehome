"""
gigo_pipeline_amantha_2.py

CLI-friendly script for the "GIGO in Practice – Data Quality Pipeline" example.

- Simulates a reasonably clean transaction dataset.
- Injects realistic "garbage" (missing values, invalid values, outliers, duplicates).
- Prints a data-quality report before and after cleaning.
- Applies a cleaning pipeline.
- Compares average transaction amount per country before vs after cleaning.
- Saves the corrupted ("dirty") dataset to data/transactions_dirty.csv
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# 1. Configuration / constants
# -----------------------------

RANDOM_SEED = 42
N_ROWS = 500

VALID_COUNTRIES: List[str] = ["US", "UK", "IN", "DE", "CA"]
VALID_PRODUCT_CATEGORIES: List[str] = ["Electronics", "Clothing", "Grocery", "Beauty"]

# Simple validation rules for the main columns
VALIDATION_RULES: Dict[str, Dict] = {
    "age": {"min": 0, "max": 120},
    "country": {"allowed": VALID_COUNTRIES},
    "product_category": {"allowed": VALID_PRODUCT_CATEGORIES},
    "transaction_amount": {"min": 0, "max": 1000},
}


# -----------------------------
# 2. Data generation
# -----------------------------

def generate_base_transactions(
    n_rows: int = N_ROWS, random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Create a "reasonably clean" synthetic transaction dataset matching the notebook.

    Columns:
      - customer_id
      - age
      - country
      - product_category
      - transaction_amount
    """
    np.random.seed(random_seed)

    customer_ids = np.random.randint(1000, 2000, size=n_rows)
    ages = np.random.randint(18, 80, size=n_rows)
    countries = np.random.choice(
        VALID_COUNTRIES,
        size=n_rows,
        p=[0.3, 0.2, 0.2, 0.15, 0.15],
    )
    product_categories = np.random.choice(
        VALID_PRODUCT_CATEGORIES,
        size=n_rows,
    )
    transaction_amounts = np.round(
        np.random.exponential(scale=50, size=n_rows) + 10, 2
    )

    base_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "country": countries,
            "product_category": product_categories,
            "transaction_amount": transaction_amounts,
        }
    )
    return base_df


# -----------------------------
# 3. Inject GIGO / corruption
# -----------------------------

def inject_garbage(df: pd.DataFrame, random_seed: int = 123) -> pd.DataFrame:
    """
    Take a clean DataFrame and inject a variety of data quality problems:
      - Missing values
      - Impossible ages
      - Invalid country / product codes
      - Negative and extreme transaction amounts
      - Duplicated rows
    """
    rng = np.random.default_rng(random_seed)
    dirty = df.copy()

    n_rows = len(dirty)
    idx = np.arange(n_rows)

    # 3.1 Random missing values
    mask_missing_country = rng.random(n_rows) < 0.08
    mask_missing_age = rng.random(n_rows) < 0.05
    mask_missing_amount = rng.random(n_rows) < 0.05
    mask_missing_product = rng.random(n_rows) < 0.03

    dirty.loc[mask_missing_country, "country"] = np.nan
    dirty.loc[mask_missing_age, "age"] = np.nan
    dirty.loc[mask_missing_amount, "transaction_amount"] = np.nan
    dirty.loc[mask_missing_product, "product_category"] = np.nan

    # 3.2 Impossible ages (negative, > 120)
    mask_weird_age = rng.random(n_rows) < 0.05
    weird_values = rng.choice([-5, -10, 130, 150], size=mask_weird_age.sum())
    dirty.loc[mask_weird_age, "age"] = weird_values

    # 3.3 Invalid country and product codes
    bad_countries = rng.choice(["XX", "Narnia", "??"], size=10)
    bad_country_indices = rng.choice(idx, size=len(bad_countries), replace=False)
    dirty.loc[bad_country_indices, "country"] = bad_countries

    bad_products = rng.choice(["Unknown", "???", "Misc"], size=10)
    bad_product_indices = rng.choice(idx, size=len(bad_products), replace=False)
    dirty.loc[bad_product_indices, "product_category"] = bad_products

    # 3.4 Negative and extreme transaction amounts
    neg_indices = rng.choice(idx, size=15, replace=False)
    dirty.loc[neg_indices, "transaction_amount"] = -rng.uniform(1, 200, size=len(neg_indices))

    extreme_indices = rng.choice(idx, size=15, replace=False)
    dirty.loc[extreme_indices, "transaction_amount"] = rng.uniform(2000, 10000, size=len(extreme_indices))

    # 3.5 Duplicate some rows
    dup_indices = rng.choice(idx, size=40, replace=False)
    duplicates = dirty.iloc[dup_indices]
    dirty = pd.concat([dirty, duplicates], ignore_index=True)

    return dirty


# -----------------------------
# 4. Data-quality report
# -----------------------------

def _invalid_mask(col: pd.Series, rules: Dict) -> pd.Series:
    """
    For a given column and its rules, return a boolean mask of invalid (non-missing) values.
    """
    mask = pd.Series(False, index=col.index)

    # Range checks
    if "min" in rules:
        mask |= (col < rules["min"]) & col.notna()
    if "max" in rules:
        mask |= (col > rules["max"]) & col.notna()

    # Categorical allowed-values check
    if "allowed" in rules:
        mask |= ~col.isin(rules["allowed"]) & col.notna()

    return mask


def data_quality_report(
    df: pd.DataFrame, validation_rules: Dict[str, Dict] = VALIDATION_RULES
) -> Tuple[pd.DataFrame, float]:
    """
    Compute a simple data quality report:
      - Missing percentage per column
      - Invalid percentage per column (based on VALIDATION_RULES)
      - Duplicate row percentage (overall)
    """
    results = []

    for col_name, rules in validation_rules.items():
        if col_name not in df.columns:
            continue

        col = df[col_name]
        missing_pct = 100 * col.isna().mean()
        invalid_mask = _invalid_mask(col, rules)
        invalid_pct = 100 * invalid_mask.mean()

        results.append(
            {
                "column": col_name,
                "missing_pct": round(missing_pct, 2),
                "invalid_pct": round(invalid_pct, 2),
            }
        )

    dq_df = pd.DataFrame(results)

    # Duplicates: compare length before vs after dropping duplicates
    if len(df) > 0:
        dup_pct = 100 * (1 - len(df.drop_duplicates()) / len(df))
    else:
        dup_pct = 0.0

    return dq_df, dup_pct


# -----------------------------
# 5. Cleaning pipeline
# -----------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a simple cleaning pipeline that:

      - Drops exact duplicate rows.
      - Sets impossible ages to NaN and clips ages to [0, 120].
      - Replaces invalid country/product codes with NaN.
      - Sets negative or extreme transaction amounts to NaN, then
        optionally caps very large but plausible values.
      - Drops rows that still have missing values in key fields.
    """
    clean = df.copy()

    # 5.1 Remove exact duplicates
    clean = clean.drop_duplicates().reset_index(drop=True)

    # 5.2 Age cleaning
    #    - Anything < 0 or > 120 is treated as missing
    clean.loc[(clean["age"] < 0) | (clean["age"] > 120), "age"] = np.nan

    # 5.3 Country + product categorical cleaning
    clean.loc[~clean["country"].isin(VALID_COUNTRIES), "country"] = np.nan
    clean.loc[~clean["product_category"].isin(VALID_PRODUCT_CATEGORIES), "product_category"] = np.nan

    # 5.4 Transaction amount cleaning
    # Negative or > 10,000 are treated as missing (garbage)
    clean.loc[(clean["transaction_amount"] < 0) | (clean["transaction_amount"] > 10000), "transaction_amount"] = np.nan

    # 5.5 Drop rows with missing in critical fields
    clean = clean.dropna(
        subset=["customer_id", "age", "country", "product_category", "transaction_amount"]
    ).reset_index(drop=True)

    # 5.6 Cap extremely large but plausible values (winsorization)
    #      (e.g., 99th percentile)
    if len(clean) > 0:
        cap_value = clean["transaction_amount"].quantile(0.99)
        clean["transaction_amount"] = np.minimum(clean["transaction_amount"], cap_value)

    return clean


# -----------------------------
# 6. Utility: metric comparison
# -----------------------------

def compare_metric_by_country(
    dirty: pd.DataFrame, clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute average transaction_amount by country before vs after cleaning.
    Returns a small summary DataFrame.
    """
    before = (
        dirty.groupby("country")["transaction_amount"]
        .mean()
        .rename("avg_amount_before")
    )
    after = (
        clean.groupby("country")["transaction_amount"]
        .mean()
        .rename("avg_amount_after")
    )

    summary = pd.concat([before, after], axis=1)
    summary["delta"] = summary["avg_amount_after"] - summary["avg_amount_before"]
    return summary


# -----------------------------
# 7. Main entry point
# -----------------------------

def main() -> None:
    print("=" * 80)
    print("GIGO in Practice – Data Quality Pipeline (Amantha / Teaching Version)")
    print("=" * 80)

    # Step 1: generate base dataset
    base_df = generate_base_transactions(N_ROWS, RANDOM_SEED)
    print(f"\nGenerated base dataset with {len(base_df)} rows.")
    print(base_df.head(), "\n")

    # Step 2: inject garbage
    dirty_df = inject_garbage(base_df, random_seed=123)
    print(f"After injecting garbage, dataset has {len(dirty_df)} rows (with duplicates).")

    # Save example dataset to disk (for the "Example scenarios/datasets" requirement)
    os.makedirs("data", exist_ok=True)
    dirty_path = os.path.join("data", "transactions_dirty.csv")
    dirty_df.to_csv(dirty_path, index=False)
    print(f"Saved dirty dataset to: {dirty_path}")

    # Step 3: data-quality report BEFORE cleaning
    print("\n--- Data Quality Report: BEFORE Cleaning ---")
    dq_before, dup_before = data_quality_report(dirty_df, VALIDATION_RULES)
    print(dq_before.to_string(index=False))
    print(f"\nDuplicate row percentage (before): {dup_before:.2f}%")

    # Step 4: cleaning pipeline
    clean_df = clean_data(dirty_df)
    print(f"\nAfter cleaning, dataset has {len(clean_df)} rows.")

    # Step 5: data-quality report AFTER cleaning
    print("\n--- Data Quality Report: AFTER Cleaning ---")
    dq_after, dup_after = data_quality_report(clean_df, VALIDATION_RULES)
    print(dq_after.to_string(index=False))
    print(f"\nDuplicate row percentage (after): {dup_after:.2f}%")

    # Step 6: GIGO moment – business metric comparison
    print("\n--- Average Transaction Amount by Country: BEFORE vs AFTER Cleaning ---")
    metric_summary = compare_metric_by_country(dirty_df, clean_df)
    # Sort countries alphabetically for readability
    metric_summary = metric_summary.sort_index()
    print(metric_summary.round(2).to_string())

    print("\nDone. This script is primarily for teaching; see the notebook for plots and more narrative.\n")


if __name__ == "__main__":
    main()
