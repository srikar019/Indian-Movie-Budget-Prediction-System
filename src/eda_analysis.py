"""
📊 Exploratory Data Analysis Module
======================================
Professional-grade EDA with statistical tests.

Generates insights like:
- "Action films have 2.3× higher average budgets than Drama films"
- "Top actors increase average budget by ₹X crores"
- "Average budget has increased steadily post-2010"

Statistical tests:
- Shapiro-Wilk normality test on budget distribution
- Kruskal-Wallis test for genre vs budget
- Spearman correlation analysis
- Budget distribution skewness analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings("ignore")


class MovieEDA:
    """Professional-grade Exploratory Data Analysis for Indian Movies."""

    def __init__(self, df):
        self.df = df.copy()
        self.insights = []
        self.stats_results = {}

    def run_full_eda(self):
        """Execute complete EDA pipeline."""
        print("=" * 70)
        print("📊 PROFESSIONAL EDA REPORT")
        print("=" * 70)

        self.budget_distribution_analysis()
        self.genre_vs_budget_analysis()
        self.actor_impact_analysis()
        self.director_impact_analysis()
        self.production_house_analysis()
        self.temporal_trend_analysis()
        self.industry_comparison()
        self.correlation_analysis()
        self.outlier_detection()

        print("\n" + "=" * 70)
        print("💡 KEY INSIGHTS SUMMARY")
        print("=" * 70)
        for i, insight in enumerate(self.insights, 1):
            print(f"   {i}. {insight}")

        return self.insights, self.stats_results

    # ──────────────────────────────────────────────────────
    def budget_distribution_analysis(self):
        """Analyze budget distribution: skewness, normality, outliers."""
        print("\n📈 Budget Distribution Analysis")
        print("─" * 50)

        budgets = self.df["budget_crores"].dropna()

        # Basic stats
        mean_b = budgets.mean()
        median_b = budgets.median()
        std_b = budgets.std()
        skew = budgets.skew()
        kurt = budgets.kurtosis()

        print(f"   Mean:    ₹{mean_b:.2f} Cr")
        print(f"   Median:  ₹{median_b:.2f} Cr")
        print(f"   Std Dev: ₹{std_b:.2f} Cr")
        print(f"   Skewness: {skew:.3f} ({'right-skewed' if skew > 0 else 'left-skewed'})")
        print(f"   Kurtosis: {kurt:.3f} ({'heavy-tailed' if kurt > 3 else 'normal-tailed'})")

        # Normality test
        if len(budgets) > 5000:
            stat, p_value = stats.normaltest(budgets.values[:5000])
            test_name = "D'Agostino"
        else:
            stat, p_value = stats.shapiro(budgets.values[:5000] if len(budgets) > 5000 else budgets.values)
            test_name = "Shapiro-Wilk"

        print(f"\n   {test_name} Normality Test:")
        print(f"   Statistic = {stat:.6f}, p-value = {p_value:.6e}")
        if p_value < 0.05:
            print(f"   → Budget is NOT normally distributed (p < 0.05)")
            self.insights.append("Budget distribution is right-skewed (most films are low-budget with a few big outliers)")
        else:
            print(f"   → Budget follows normal distribution")

        # Percentile analysis
        low_budget_pct = (budgets < 50).sum() / len(budgets) * 100
        high_budget_pct = (budgets >= 200).sum() / len(budgets) * 100
        self.insights.append(f"{low_budget_pct:.0f}% of movies have budgets under ₹50 Cr, "
                             f"only {high_budget_pct:.0f}% exceed ₹200 Cr")

        self.stats_results["budget_distribution"] = {
            "mean": round(mean_b, 2), "median": round(median_b, 2),
            "std": round(std_b, 2), "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "normality_test": test_name,
            "normality_p_value": float(p_value),
            "is_normal": p_value >= 0.05,
            "low_budget_pct": round(low_budget_pct, 1),
            "high_budget_pct": round(high_budget_pct, 1),
        }

    # ──────────────────────────────────────────────────────
    def genre_vs_budget_analysis(self):
        """Analyze which genres cost more."""
        print("\n🎭 Genre vs Budget Analysis")
        print("─" * 50)

        genre_stats = self.df.groupby("genre")["budget_crores"].agg(
            ["mean", "median", "count", "std"]
        ).sort_values("mean", ascending=False)

        # Filter genres with meaningful sample sizes
        genre_stats = genre_stats[genre_stats["count"] >= 5]

        print("\n   Genre Budget Rankings (min 5 movies):")
        for genre, row in genre_stats.iterrows():
            bar = "█" * int(row["mean"] / genre_stats["mean"].max() * 25)
            print(f"   {genre:15s} | Avg: ₹{row['mean']:7.2f} Cr | n={int(row['count']):4d} | {bar}")

        # Compare most vs least expensive genre
        if len(genre_stats) >= 2:
            most_exp = genre_stats.index[0]
            least_exp = genre_stats.index[-1]
            ratio = genre_stats.loc[most_exp, "mean"] / genre_stats.loc[least_exp, "mean"]
            self.insights.append(
                f"{most_exp} films have {ratio:.1f}× higher average budgets than {least_exp} films"
            )

        # Kruskal-Wallis test (non-parametric ANOVA)
        groups = [group["budget_crores"].values for _, group in self.df.groupby("genre")
                  if len(group) >= 5]
        if len(groups) >= 2:
            stat, p_value = stats.kruskal(*groups)
            print(f"\n   Kruskal-Wallis Test (genre → budget):")
            print(f"   H = {stat:.4f}, p = {p_value:.2e}")
            if p_value < 0.05:
                self.insights.append("Genre significantly affects budget (Kruskal-Wallis p < 0.05)")

            self.stats_results["genre_budget_test"] = {
                "test": "Kruskal-Wallis",
                "statistic": round(stat, 4),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }

    # ──────────────────────────────────────────────────────
    def actor_impact_analysis(self):
        """Analyze how top actors affect budgets."""
        print("\n⭐ Actor Impact Analysis")
        print("─" * 50)

        actor_stats = self.df.groupby("lead_actor").agg(
            avg_budget=("budget_crores", "mean"),
            avg_box_office=("box_office_crores", "mean"),
            count=("movie_id", "count"),
        ).reset_index()
        actor_stats = actor_stats[actor_stats["count"] >= 3].sort_values(
            "avg_budget", ascending=False
        )

        # Top 10 actors by budget
        print("\n   Top 10 Actors by Average Budget:")
        for _, row in actor_stats.head(10).iterrows():
            print(f"   {row['lead_actor']:25s} | Avg Budget: ₹{row['avg_budget']:7.2f} Cr "
                  f"| Avg BO: ₹{row['avg_box_office']:7.2f} Cr | Movies: {int(row['count'])}")

        # Compare top actors vs rest
        top_actor_names = actor_stats.head(10)["lead_actor"].values
        top_actor_avg = self.df[self.df["lead_actor"].isin(top_actor_names)]["budget_crores"].mean()
        others_avg = self.df[~self.df["lead_actor"].isin(top_actor_names)]["budget_crores"].mean()
        diff = top_actor_avg - others_avg
        self.insights.append(
            f"Top 10 actors increase average budget by ₹{diff:.1f} Cr compared to other actors"
        )

    # ──────────────────────────────────────────────────────
    def director_impact_analysis(self):
        """Analyze director impact on budgets."""
        print("\n🎬 Director Impact Analysis")
        print("─" * 50)

        dir_stats = self.df.groupby("director").agg(
            avg_budget=("budget_crores", "mean"),
            hit_count=("verdict", lambda x: (x.isin(["Blockbuster", "Super Hit", "Hit"])).sum()),
            total=("movie_id", "count"),
        ).reset_index()
        dir_stats = dir_stats[dir_stats["total"] >= 3]
        dir_stats["hit_rate"] = (dir_stats["hit_count"] / dir_stats["total"] * 100).round(1)
        dir_stats = dir_stats.sort_values("avg_budget", ascending=False)

        print("\n   Top 10 Directors by Average Budget:")
        for _, row in dir_stats.head(10).iterrows():
            print(f"   {row['director']:25s} | Avg Budget: ₹{row['avg_budget']:7.2f} Cr "
                  f"| Hit Rate: {row['hit_rate']:5.1f}% | Movies: {int(row['total'])}")

        # Correlation between director hit rate and budget
        if len(dir_stats) > 10:
            corr, p_val = stats.spearmanr(dir_stats["hit_rate"], dir_stats["avg_budget"])
            self.insights.append(
                f"Director hit rate correlates with budget (Spearman ρ = {corr:.3f}, p = {p_val:.4f})"
            )

    # ──────────────────────────────────────────────────────
    def production_house_analysis(self):
        """Analyze production house budgets."""
        print("\n🏢 Production House Analysis")
        print("─" * 50)

        if "production_house" not in self.df.columns:
            print("   ⚠️ No production_house column, skipping")
            return

        ph_stats = self.df.groupby("production_house").agg(
            avg_budget=("budget_crores", "mean"),
            count=("movie_id", "count"),
        ).reset_index()
        ph_stats = ph_stats[ph_stats["count"] >= 5].sort_values("avg_budget", ascending=False)

        print("\n   Top 10 Production Houses by Budget:")
        for _, row in ph_stats.head(10).iterrows():
            print(f"   {row['production_house']:35s} | Avg: ₹{row['avg_budget']:7.2f} Cr | n={int(row['count'])}")

        if len(ph_stats) >= 5:
            big_studios = ph_stats.head(5)["production_house"].values
            big_avg = self.df[self.df["production_house"].isin(big_studios)]["budget_crores"].mean()
            small_avg = self.df[~self.df["production_house"].isin(big_studios)]["budget_crores"].mean()
            ratio = big_avg / small_avg if small_avg > 0 else 0
            self.insights.append(
                f"Top 5 production houses have {ratio:.1f}× higher average budgets "
                f"(₹{big_avg:.0f} Cr vs ₹{small_avg:.0f} Cr)"
            )

    # ──────────────────────────────────────────────────────
    def temporal_trend_analysis(self):
        """Budget trends over time."""
        print("\n📅 Temporal Trend Analysis")
        print("─" * 50)

        yearly = self.df.groupby("year")["budget_crores"].agg(["mean", "median", "count"])
        yearly = yearly[yearly["count"] >= 3]

        if len(yearly) > 5:
            # Linear regression on yearly average
            years = yearly.index.values
            means = yearly["mean"].values
            slope, intercept, r_val, p_val, std_err = stats.linregress(years, means)

            print(f"   Trend: ₹{slope:.2f} Cr/year increase")
            print(f"   R² = {r_val**2:.4f}")
            self.insights.append(
                f"Average budget increases by ₹{slope:.2f} Cr per year (R² = {r_val**2:.3f})"
            )

            # Post-2010 acceleration
            post_2010 = yearly[yearly.index >= 2010]
            pre_2010 = yearly[yearly.index < 2010]
            if len(post_2010) > 2 and len(pre_2010) > 2:
                post_avg = post_2010["mean"].mean()
                pre_avg = pre_2010["mean"].mean()
                self.insights.append(
                    f"Post-2010 average budget (₹{post_avg:.0f} Cr) is "
                    f"{post_avg/pre_avg:.1f}× higher than pre-2010 (₹{pre_avg:.0f} Cr)"
                )

    # ──────────────────────────────────────────────────────
    def industry_comparison(self):
        """Compare industries."""
        print("\n🌍 Industry Comparison")
        print("─" * 50)

        ind_stats = self.df.groupby("industry").agg(
            avg_budget=("budget_crores", "mean"),
            avg_bo=("box_office_crores", "mean"),
            count=("movie_id", "count"),
        ).sort_values("avg_budget", ascending=False)

        for industry, row in ind_stats.iterrows():
            roi = (row["avg_bo"] - row["avg_budget"]) / row["avg_budget"] * 100
            print(f"   {industry:15s} | Avg Budget: ₹{row['avg_budget']:7.2f} Cr | "
                  f"Avg BO: ₹{row['avg_bo']:7.2f} Cr | ROI: {roi:+.1f}% | n={int(row['count'])}")

    # ──────────────────────────────────────────────────────
    def correlation_analysis(self):
        """Spearman correlation with budget."""
        print("\n📊 Correlation Analysis (with Budget)")
        print("─" * 50)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        budget_corrs = []

        for col in numeric_cols:
            if col != "budget_crores" and self.df[col].nunique() > 2:
                corr, p_val = stats.spearmanr(
                    self.df[col].dropna(), self.df["budget_crores"].iloc[:len(self.df[col].dropna())]
                )
                if not np.isnan(corr):
                    budget_corrs.append({
                        "Feature": col, "Correlation": round(corr, 4),
                        "p_value": p_val, "Significant": p_val < 0.05
                    })

        corr_df = pd.DataFrame(budget_corrs).sort_values("Correlation", key=abs, ascending=False)

        print("\n   Top 15 Features Correlated with Budget:")
        for _, row in corr_df.head(15).iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"   {row['Feature']:35s} | ρ = {row['Correlation']:+.4f} {sig}")

        # Top insight
        if len(corr_df) > 0:
            top_feat = corr_df.iloc[0]
            self.insights.append(
                f"Strongest budget predictor: '{top_feat['Feature']}' "
                f"(Spearman ρ = {top_feat['Correlation']:+.4f})"
            )

        self.stats_results["correlations"] = corr_df.head(20).to_dict(orient="records")

    # ──────────────────────────────────────────────────────
    def outlier_detection(self):
        """Detect budget outliers using IQR method."""
        print("\n🔍 Outlier Detection")
        print("─" * 50)

        Q1 = self.df["budget_crores"].quantile(0.25)
        Q3 = self.df["budget_crores"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = self.df[
            (self.df["budget_crores"] < lower) | (self.df["budget_crores"] > upper)
        ]

        print(f"   IQR: ₹{IQR:.2f} Cr")
        print(f"   Outlier bounds: ₹{max(0, lower):.2f} - ₹{upper:.2f} Cr")
        print(f"   Outliers detected: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")

        if len(outliers) > 0 and "title" in outliers.columns:
            top_outliers = outliers.nlargest(5, "budget_crores")
            print("\n   Top 5 Budget Outliers:")
            for _, row in top_outliers.iterrows():
                print(f"   • {row['title']} ({row['year']}) — ₹{row['budget_crores']:.2f} Cr")


if __name__ == "__main__":
    from data_cleaning import run_full_cleaning
    from feature_engineering import create_all_features

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")
    df = run_full_cleaning(raw_path)
    df = create_all_features(df)

    eda = MovieEDA(df)
    insights, stats = eda.run_full_eda()

    # Save insights
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "eda_insights.json"), "w") as f:
        json.dump({"insights": insights, "statistics": stats}, f, indent=2, default=str)
    print(f"\n💾 EDA results saved to data/eda_insights.json")
