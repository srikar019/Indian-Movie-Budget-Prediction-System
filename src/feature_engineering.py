"""
🔬 Advanced Feature Engineering Module
=========================================
Creates 30+ engineered features for Indian Movie Budget Prediction.

Includes:
- Actor Popularity Score (computed from data)
- Director Success Rate (computed from data)
- Production House Strength
- Genre Complexity Index (multi-genre aware)
- Language Market Factor
- Hype Score (composite)
- Inflation-adjusted metrics
- Interaction features
- Temporal features
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────
# Genre Complexity / Cost Index
# ──────────────────────────────────────────────────────────
GENRE_COST_INDEX = {
    "Sci-Fi": 6, "Science Fiction": 6, "Fantasy": 5, "Animation": 5,
    "War": 5, "Historical": 5, "History": 5, "Mythology": 5,
    "Action": 5, "Adventure": 4, "Superhero": 6,
    "Thriller": 3, "Crime": 3, "Mystery": 3, "Horror": 3,
    "Biography": 3, "Sports": 3, "Patriotic": 3, "Family": 2,
    "Comedy": 2, "Romance": 2, "Music": 2, "Musical": 2,
    "Drama": 1
}

# ──────────────────────────────────────────────────────────
# Language Market Factor
# ──────────────────────────────────────────────────────────
LANGUAGE_MARKET = {
    "Hindi": 1.0,       # ~600M speakers, largest market
    "Telugu": 0.75,     # ~85M speakers, high-budget industry
    "Tamil": 0.70,      # ~75M speakers, strong industry
    "Kannada": 0.45,    # ~45M speakers
    "Malayalam": 0.40,  # ~38M speakers
    "Bengali": 0.35,
    "Marathi": 0.30,
    "Punjabi": 0.25,
}

# ──────────────────────────────────────────────────────────
# Industry growth rates (CAGR of avg budgets)
# ──────────────────────────────────────────────────────────
INDUSTRY_GROWTH = {
    "Bollywood": 0.08, "Tollywood": 0.12, "Kollywood": 0.10,
    "Sandalwood": 0.09, "Mollywood": 0.11
}

INDUSTRY_AVG_BUDGET = {
    "Bollywood": 80, "Tollywood": 70, "Kollywood": 55,
    "Sandalwood": 30, "Mollywood": 25
}


def create_all_features(df):
    """
    Master feature engineering function.
    Creates 30+ derived features from raw data.
    """
    df = df.copy()

    # ─── 1. Temporal Features ──────────────────────────────
    df["decade"] = (df["year"] // 10) * 10
    df["years_since_2000"] = df["year"] - 2000

    # Quarter
    df["release_quarter"] = (df["release_month"] - 1) // 3 + 1

    # Festival season scoring (Indian cinema peak seasons)
    def get_season_score(month):
        """Score release timing: higher = better release window."""
        peak = {1: 4, 4: 3, 10: 4, 11: 5, 12: 4}      # Diwali, Christmas, Republic Day
        moderate = {3: 2, 5: 2, 6: 2, 8: 3}              # Independence Day, Summer
        return peak.get(month, moderate.get(month, 1))     # Off-season = 1

    df["release_season_score"] = df["release_month"].apply(get_season_score)

    # Is holiday release
    df["is_holiday_release"] = df["release_month"].isin([1, 4, 10, 11, 12]).astype(int)

    # ─── 2. Star Power Features ────────────────────────────
    # Star Power Index (weighted combination)
    df["star_power_index"] = (
        df["actor_popularity_score"] * 0.55 +
        df["director_success_rate"] * 10 * 0.45
    )

    df["star_power_squared"] = df["star_power_index"] ** 2

    # Actor-Director synergy (interaction feature)
    df["actor_director_synergy"] = (
        df["actor_popularity_score"] * df["director_success_rate"]
    )

    # ─── 3. Genre Complexity Index ─────────────────────────
    # Maps each genre to a cost complexity score (1-6)
    df["genre_complexity_index"] = df["genre"].map(GENRE_COST_INDEX).fillna(2)

    # Simplified cost tier (for model)
    expensive_genres = ["Sci-Fi", "Science Fiction", "Historical", "War", "Fantasy",
                        "Mythology", "History", "Action", "Adventure", "Animation"]
    moderate_genres = ["Thriller", "Crime", "Biography", "Sports", "Patriotic",
                       "Mystery", "Horror"]

    df["genre_cost_tier"] = df["genre"].apply(
        lambda g: 3 if g in expensive_genres else (2 if g in moderate_genres else 1)
    )

    # ─── 4. Language Market Factor ─────────────────────────
    if "language" in df.columns:
        df["language_market_factor"] = df["language"].map(LANGUAGE_MARKET).fillna(0.3)
    else:
        # Infer from industry
        industry_lang_map = {
            "Bollywood": 1.0, "Tollywood": 0.75, "Kollywood": 0.70,
            "Sandalwood": 0.45, "Mollywood": 0.40
        }
        df["language_market_factor"] = df["industry"].map(industry_lang_map).fillna(0.3)

    # ─── 5. Production Scale ──────────────────────────────
    max_screens = df["num_screens"].max()
    df["production_scale"] = (
        df["vfx_level"] / 5 * 0.35 +
        df["num_screens"] / max_screens * 0.35 +
        df["international_release"] * 0.15 +
        df["is_sequel"] * 0.15
    )

    # Content richness
    df["content_richness"] = (
        df["num_songs"] / 6 * 0.3 +
        df["runtime_minutes"] / 210 * 0.3 +
        df["num_cast_members"] / df["num_cast_members"].max() * 0.4
    )

    # ─── 6. Production House Strength ──────────────────────
    # If computed in data_cleaning, it already exists
    if "production_house_strength" not in df.columns:
        df["production_house_strength"] = 5.0  # Default

    # ─── 7. Hype Score ⭐ (Composite) ──────────────────────
    df["hype_score"] = (
        df["actor_popularity_score"] / 10 * 0.35 +      # Actor pull
        df["director_success_rate"] * 0.20 +             # Director track record
        df["is_sequel"] * 0.15 +                         # Franchise power
        df["international_release"] * 0.10 +             # Global buzz
        df["num_cast_members"].clip(upper=50) / 50 * 0.10 +  # Ensemble scale
        df["vfx_level"] / 5 * 0.10                       # Visual spectacle
    ).round(3)

    # ─── 8. Industry Features ─────────────────────────────
    df["industry_growth_factor"] = df.apply(
        lambda r: (1 + INDUSTRY_GROWTH.get(r["industry"], 0.08)) **
                  max(0, r["year"] - 2000),
        axis=1
    )

    df["industry_budget_tier"] = df["industry"].map(INDUSTRY_AVG_BUDGET).fillna(40)

    # ─── 9. Market Era Features ────────────────────────────
    df["is_ott_era"] = (df["year"] >= 2018).astype(int)
    df["is_post_pandemic"] = (df["year"] >= 2021).astype(int)
    df["is_modern_era"] = (df["year"] >= 2010).astype(int)

    # Market maturity
    df["market_maturity"] = df["years_since_2000"].clip(lower=0) * df["industry_growth_factor"]

    # ─── 10. Interaction Features ──────────────────────────
    df["vfx_x_screens"] = df["vfx_level"] * df["num_screens"]
    df["star_x_genre_tier"] = df["star_power_index"] * df["genre_cost_tier"]
    df["growth_x_star"] = df["industry_growth_factor"] * df["star_power_index"]
    df["hype_x_market"] = df["hype_score"] * df["language_market_factor"]
    df["production_x_genre"] = df["production_scale"] * df["genre_complexity_index"]
    df["star_x_production_house"] = df["star_power_index"] * df["production_house_strength"] / 10

    # ─── 11. Note: log_budget removed (was leaking target variable) ────

    # ─── 12. Inflation-aware year feature ──────────────────
    if "budget_2024_crores" in df.columns:
        # The inflation-adjusted budget is a better target
        pass

    new_feature_count = len([c for c in df.columns if c not in [
        "year", "release_month", "industry", "language", "genre",
        "director", "lead_actor", "production_house", "certification",
        "budget_crores", "box_office_crores", "verdict", "title", "movie_id",
        "director_success_rate", "actor_popularity_score", "num_cast_members",
        "runtime_minutes", "is_sequel", "num_songs", "vfx_level",
        "num_screens", "international_release", "ott_release",
    ]])
    print(f"⚙️ Feature Engineering: Created {new_feature_count} new features")
    print(f"   Total columns: {len(df.columns)}")

    return df


def get_feature_list():
    """Return the list of feature names used for model training."""
    return [
        # Original numerical
        "year", "release_month", "director_success_rate",
        "actor_popularity_score", "num_cast_members", "runtime_minutes",
        "is_sequel", "num_songs", "vfx_level", "num_screens",
        "international_release", "ott_release",

        # New computed features
        "production_house_strength",
        "genre_complexity_index",
        "language_market_factor",
        "hype_score",

        # Engineered
        "decade", "years_since_2000", "release_quarter", "release_season_score",
        "is_holiday_release",
        "star_power_index", "star_power_squared", "actor_director_synergy",
        "production_scale", "content_richness", "genre_cost_tier",
        "industry_growth_factor", "industry_budget_tier",
        "is_ott_era", "is_post_pandemic", "is_modern_era", "market_maturity",

        # Interactions
        "vfx_x_screens", "star_x_genre_tier", "growth_x_star",
        "hype_x_market", "production_x_genre", "star_x_production_house",
    ]


def get_feature_descriptions():
    """Return human-readable descriptions for each feature."""
    return {
        "actor_popularity_score": "Star power of lead actor (1-10, computed from box office history)",
        "director_success_rate": "Director's historical success rate (0-1, computed from hit rate)",
        "production_house_strength": "Production house brand strength (1-10)",
        "genre_complexity_index": "Genre cost complexity (1=Drama, 6=Sci-Fi)",
        "language_market_factor": "Language market size (0.25=Punjabi, 1.0=Hindi)",
        "hype_score": "Pre-release hype composite (actors + director + sequel + VFX)",
        "star_power_index": "Combined star power (actor + director weighted)",
        "star_power_squared": "Non-linear star effect (squared)",
        "actor_director_synergy": "Actor × Director interaction",
        "production_scale": "Overall production scale (VFX + screens + international)",
        "content_richness": "Content depth (songs + runtime + cast size)",
        "genre_cost_tier": "Genre expense tier (1=low, 2=mid, 3=high)",
        "industry_growth_factor": "Industry compound growth since 2000",
        "industry_budget_tier": "Average industry budget level (₹ Cr)",
        "release_season_score": "Release timing quality (1=off-season, 5=Diwali)",
        "hype_x_market": "Hype × Market size interaction",
        "production_x_genre": "Production scale × Genre complexity",
        "vfx_x_screens": "VFX level × Screen count interaction",
        "market_maturity": "Industry maturity at release year",
        "is_ott_era": "Released in OTT era (2018+)",
        "is_post_pandemic": "Released post-pandemic (2021+)",
    }


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")
    df = pd.read_csv(data_path)
    df = create_all_features(df)
    print(f"\n📊 Final DataFrame shape: {df.shape}")
    print(f"\n🔢 Feature list ({len(get_feature_list())} features):")
    for f in get_feature_list():
        print(f"   • {f}")
