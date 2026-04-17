"""
🧹 Advanced Data Cleaning & Feature Computation Module
=========================================================
Deduplicates movies, computes REAL actor/director/production scores
from historical performance data, and adds inflation-adjusted budgets.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────
# Indian CPI-based inflation index (base year 2024 = 100)
# Source: RBI/MOSPI approximate CPI-IW index
# ──────────────────────────────────────────────────────────
INFLATION_INDEX = {
    1950: 1.2, 1951: 1.3, 1955: 1.5, 1957: 1.6, 1960: 1.8,
    1963: 2.0, 1964: 2.1, 1968: 2.5, 1969: 2.6, 1970: 2.7,
    1971: 2.8, 1972: 3.0, 1973: 3.3, 1974: 3.8, 1975: 3.6,
    1976: 3.5, 1977: 3.7, 1978: 3.8, 1979: 4.0, 1980: 4.5,
    1981: 5.0, 1982: 5.3, 1983: 5.7, 1984: 6.0, 1985: 6.3,
    1986: 6.8, 1987: 7.2, 1988: 7.8, 1989: 8.2, 1990: 9.0,
    1991: 10.2, 1992: 11.3, 1993: 12.0, 1994: 13.2, 1995: 14.5,
    1996: 15.8, 1997: 16.5, 1998: 18.0, 1999: 18.8, 2000: 19.5,
    2001: 20.3, 2002: 21.0, 2003: 21.8, 2004: 22.7, 2005: 23.5,
    2006: 25.0, 2007: 26.5, 2008: 29.0, 2009: 31.5, 2010: 35.0,
    2011: 38.0, 2012: 41.5, 2013: 45.5, 2014: 48.5, 2015: 51.0,
    2016: 53.5, 2017: 55.5, 2018: 58.0, 2019: 61.0, 2020: 65.0,
    2021: 68.5, 2022: 73.0, 2023: 78.0, 2024: 82.0, 2025: 86.0,
    2026: 90.0, 2027: 94.0
}

CPI_2024 = INFLATION_INDEX[2024]


def get_cpi(year):
    """Get CPI index for a year (interpolate if missing)."""
    if year in INFLATION_INDEX:
        return INFLATION_INDEX[year]
    # Interpolate between known years
    known_years = sorted(INFLATION_INDEX.keys())
    for i in range(len(known_years) - 1):
        if known_years[i] <= year <= known_years[i + 1]:
            y1, y2 = known_years[i], known_years[i + 1]
            c1, c2 = INFLATION_INDEX[y1], INFLATION_INDEX[y2]
            return c1 + (c2 - c1) * (year - y1) / (y2 - y1)
    return INFLATION_INDEX[known_years[-1]]


def deduplicate_movies(df):
    """
    Remove duplicate movies by title.
    Keep the entry with the most complete data (fewest NaN values).
    """
    print("🧹 Deduplicating movies...")
    initial_count = len(df)

    # If title column exists, deduplicate by title
    if "title" in df.columns:
        # For each duplicated title, keep the row with the most data
        df["_completeness"] = df.notna().sum(axis=1)
        df = df.sort_values("_completeness", ascending=False)
        df = df.drop_duplicates(subset=["title"], keep="first")
        df = df.drop(columns=["_completeness"])
    elif "movie_name" in df.columns:
        df["_completeness"] = df.notna().sum(axis=1)
        df = df.sort_values("_completeness", ascending=False)
        df = df.drop_duplicates(subset=["movie_name"], keep="first")
        df = df.drop(columns=["_completeness"])
    else:
        # Fallback: deduplicate by movie_id
        df = df.drop_duplicates(subset=["movie_id"], keep="first")

    removed = initial_count - len(df)
    print(f"   ✅ Removed {removed} duplicates → {len(df)} unique movies")
    return df.reset_index(drop=True)


def compute_actor_popularity(df):
    """
    Compute actor popularity scores WITHOUT data leakage.
    
    IMPORTANT: We do NOT use box_office_crores or verdict here because
    those are outcome variables correlated with budget. Using them leaks
    future/outcome information into features.
    
    Instead we use:
    - Number of movies (career volume)
    - Average budget worked with (market value signal)
    - Max budget worked with (star-tier signal)
    - Career span in years (longevity)
    
    Scale: 1.0 (unknown) to 10.0 (superstar)
    """
    print("⭐ Computing actor popularity scores (leakage-free)...")

    actor_stats = df.groupby("lead_actor").agg(
        avg_budget=("budget_crores", "mean"),
        max_budget=("budget_crores", "max"),
        total_movies=("movie_id", "count"),
        year_min=("year", "min"),
        year_max=("year", "max"),
    ).reset_index()

    # Career span (longevity)
    actor_stats["career_span"] = (actor_stats["year_max"] - actor_stats["year_min"]).clip(lower=0)

    # Normalize each component to [0, 1]
    for col in ["avg_budget", "max_budget", "total_movies", "career_span"]:
        max_val = actor_stats[col].max()
        if max_val > 0:
            actor_stats[f"{col}_norm"] = actor_stats[col] / max_val
        else:
            actor_stats[f"{col}_norm"] = 0

    # Weighted composite score → scale to [1, 10]
    actor_stats["computed_popularity"] = (
        actor_stats["avg_budget_norm"] * 0.35 +
        actor_stats["max_budget_norm"] * 0.25 +
        actor_stats["total_movies_norm"] * 0.25 +
        actor_stats["career_span_norm"] * 0.15
    )
    actor_stats["computed_popularity"] = (
        1 + actor_stats["computed_popularity"] * 9
    ).clip(1, 10).round(2)

    # Map back to main dataframe
    actor_map = actor_stats.set_index("lead_actor")["computed_popularity"]
    df["actor_popularity_score"] = df["lead_actor"].map(actor_map).fillna(1.0)

    top_10 = actor_stats.nlargest(10, "computed_popularity")[
        ["lead_actor", "computed_popularity", "total_movies", "avg_budget"]
    ]
    print(f"   ✅ Computed scores for {len(actor_stats)} actors")
    print(f"   🌟 Top actors: {top_10['lead_actor'].tolist()[:5]}")

    return df


def compute_director_success_rate(df):
    """
    Compute director success rates WITHOUT data leakage.
    
    IMPORTANT: We do NOT use box_office_crores or verdict here because
    those are outcome variables. Using them leaks outcome information.
    
    Instead we use:
    - Number of movies directed (experience)
    - Average budget entrusted (industry trust signal)
    - Max budget entrusted (big-budget capability)
    - Career span (longevity)
    
    Scale: 0.0 to 1.0
    """
    print("🎬 Computing director success rates (leakage-free)...")

    dir_stats = df.groupby("director").agg(
        avg_budget=("budget_crores", "mean"),
        max_budget=("budget_crores", "max"),
        total_movies=("movie_id", "count"),
        year_min=("year", "min"),
        year_max=("year", "max"),
    ).reset_index()

    # Career span
    dir_stats["career_span"] = (dir_stats["year_max"] - dir_stats["year_min"]).clip(lower=0)

    # Normalize
    max_budget = dir_stats["avg_budget"].max()
    max_budget_single = dir_stats["max_budget"].max()
    max_movies = dir_stats["total_movies"].max()
    max_span = dir_stats["career_span"].max()
    dir_stats["budget_norm"] = dir_stats["avg_budget"] / max_budget if max_budget > 0 else 0
    dir_stats["max_budget_norm"] = dir_stats["max_budget"] / max_budget_single if max_budget_single > 0 else 0
    dir_stats["exp_norm"] = dir_stats["total_movies"] / max_movies if max_movies > 0 else 0
    dir_stats["span_norm"] = dir_stats["career_span"] / max_span if max_span > 0 else 0

    # Composite success rate [0, 1]
    dir_stats["computed_success_rate"] = (
        dir_stats["budget_norm"] * 0.35 +
        dir_stats["max_budget_norm"] * 0.25 +
        dir_stats["exp_norm"] * 0.25 +
        dir_stats["span_norm"] * 0.15
    ).clip(0, 1).round(3)

    # Map back
    dir_map = dir_stats.set_index("director")["computed_success_rate"]
    df["director_success_rate"] = df["director"].map(dir_map).fillna(0.05)

    top_dirs = dir_stats.nlargest(10, "computed_success_rate")[
        ["director", "computed_success_rate", "total_movies", "avg_budget"]
    ]
    print(f"   ✅ Computed rates for {len(dir_stats)} directors")
    print(f"   🌟 Top directors: {top_dirs['director'].tolist()[:5]}")

    return df


def compute_production_house_strength(df):
    """
    Compute production house strength WITHOUT data leakage.
    
    Uses only budget-side signals (NOT box_office or verdict):
    - Average budget (financial capacity)
    - Max budget (peak investment)
    - Number of movies (industry presence)
    
    Scale: 1 (small) to 10 (mega studio)
    """
    print("🏢 Computing production house strength (leakage-free)...")

    if "production_house" not in df.columns:
        print("   ⚠️ No production_house column found, skipping.")
        df["production_house_strength"] = 5.0
        return df

    ph_stats = df.groupby("production_house").agg(
        avg_budget=("budget_crores", "mean"),
        max_budget=("budget_crores", "max"),
        total_movies=("movie_id", "count"),
    ).reset_index()

    # Normalize
    for col in ["avg_budget", "max_budget", "total_movies"]:
        mx = ph_stats[col].max()
        ph_stats[f"{col}_norm"] = ph_stats[col] / mx if mx > 0 else 0

    ph_stats["strength"] = (
        ph_stats["avg_budget_norm"] * 0.40 +
        ph_stats["max_budget_norm"] * 0.30 +
        ph_stats["total_movies_norm"] * 0.30
    )
    ph_stats["strength"] = (1 + ph_stats["strength"] * 9).clip(1, 10).round(2)

    ph_map = ph_stats.set_index("production_house")["strength"]
    df["production_house_strength"] = df["production_house"].map(ph_map).fillna(3.0)

    print(f"   ✅ Computed for {len(ph_stats)} production houses")
    return df


def add_inflation_adjusted_budget(df):
    """
    Add inflation-adjusted budget in 2024 ₹ Crores.
    This normalizes budgets across decades for fair comparison.
    """
    print("💹 Adding inflation-adjusted budgets (base year: 2024)...")

    df["cpi_index"] = df["year"].apply(get_cpi)
    df["budget_2024_crores"] = (
        df["budget_crores"] * (CPI_2024 / df["cpi_index"])
    ).round(2)

    df["box_office_2024_crores"] = (
        df["box_office_crores"] * (CPI_2024 / df["cpi_index"])
    ).round(2)

    print(f"   ✅ Added budget_2024_crores & box_office_2024_crores")
    return df


def add_roi_and_profit(df):
    """
    Add Return on Investment (ROI) and profit/loss columns.
    ROI = (Box Office - Budget) / Budget × 100
    """
    print("📊 Computing ROI and profit/loss...")

    df["profit_loss_crores"] = (df["box_office_crores"] - df["budget_crores"]).round(2)
    df["roi_percentage"] = np.where(
        df["budget_crores"] > 0,
        ((df["box_office_crores"] - df["budget_crores"]) / df["budget_crores"] * 100).round(2),
        0
    )

    # Profitability classification
    def classify_profit(roi):
        if roi >= 200:
            return "Mega Profitable"
        elif roi >= 100:
            return "Highly Profitable"
        elif roi >= 25:
            return "Profitable"
        elif roi >= -25:
            return "Break Even"
        elif roi >= -50:
            return "Loss"
        else:
            return "Disaster"

    df["profitability"] = df["roi_percentage"].apply(classify_profit)

    profitable = (df["roi_percentage"] > 0).sum()
    print(f"   ✅ {profitable}/{len(df)} movies were profitable ({profitable/len(df)*100:.1f}%)")

    return df


def run_full_cleaning(raw_data_path, save_path=None):
    """
    Execute the complete data cleaning pipeline.
    Returns the cleaned, enriched DataFrame.
    """
    print("=" * 70)
    print("🚀 ADVANCED DATA CLEANING PIPELINE")
    print("=" * 70)

    # Load
    df = pd.read_csv(raw_data_path)
    print(f"\n📂 Loaded {len(df)} raw records")

    # 1. Deduplicate
    df = deduplicate_movies(df)

    # 2. Compute real actor scores (replaces the constant 3.0)
    df = compute_actor_popularity(df)

    # 3. Compute real director success rates (replaces the constant 0.3)
    df = compute_director_success_rate(df)

    # 4. Production house strength
    df = compute_production_house_strength(df)

    # 5. Inflation-adjusted budgets
    df = add_inflation_adjusted_budget(df)

    # 6. ROI and profit/loss
    df = add_roi_and_profit(df)

    # 7. Remove remaining bad rows
    df = df[df["budget_crores"] > 0].copy()
    df = df[df["runtime_minutes"] >= 20].copy()

    print(f"\n✅ Final cleaned dataset: {len(df)} movies")
    print(f"   Columns: {len(df.columns)}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"💾 Saved to: {save_path}")

    return df


if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")
    save_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_enriched_data.csv")
    df = run_full_cleaning(raw_path, save_path)

    print("\n📊 Sample of computed scores:")
    print(df[["title", "lead_actor", "actor_popularity_score",
              "director", "director_success_rate",
              "budget_crores", "budget_2024_crores", "roi_percentage"]].head(20).to_string())
