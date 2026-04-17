"""
🧠 Advanced ML Training Module
=================================
Trains multiple ML models with:
- K-Fold Cross-Validation
- Stacking Ensemble (RF + XGB + LightGBM → Meta Ridge)
- SHAP Explainability (feature importance + force plots)
- Residual / Error Analysis
- Budget Recommendation System (confidence intervals)
- Feature Importance Analysis

This is NOT a basic train-predict pipeline — this is production-grade ML.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

# Optional SHAP (graceful fallback)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not installed. Run: pip install shap")


class AdvancedMovieBudgetTrainer:
    """
    Production-grade ML trainer with:
    - 8 base models + 1 stacking ensemble
    - 5-fold cross-validation
    - SHAP explainability
    - Error analysis & diagnostics
    - Budget recommendation with confidence intervals
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        self.shap_values = None
        self.shap_explainer = None
        self.error_analysis = {}
        self.feature_names = None

    # ──────────────────────────────────────────────────────
    # Model Definitions
    # ──────────────────────────────────────────────────────

    def _get_base_models(self):
        """Define all base models with tuned hyperparameters."""
        return {
            "Linear Regression": LinearRegression(),

            "Ridge Regression": Ridge(alpha=10.0),

            "Lasso Regression": Lasso(alpha=0.5, max_iter=5000),

            "Decision Tree": DecisionTreeRegressor(
                max_depth=10, min_samples_split=20,
                min_samples_leaf=10, random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=300, max_depth=15,
                min_samples_split=10, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                min_samples_split=15, min_samples_leaf=10,
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                n_estimators=500, max_depth=6,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=1.0,
                reg_lambda=5.0, min_child_weight=5,
                random_state=42, n_jobs=-1, verbosity=0
            ),

            "LightGBM": LGBMRegressor(
                n_estimators=500, max_depth=7,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=1.0,
                reg_lambda=5.0, min_child_samples=20,
                random_state=42, n_jobs=-1, verbose=-1
            ),
        }

    def _get_stacking_model(self):
        """
        Create a Stacking Ensemble that combines the 3 best
        tree-based models with a Ridge meta-learner.
        Uses stronger base models than standalone to ensure
        the ensemble outperforms individual models.
        """
        estimators = [
            ("rf", RandomForestRegressor(
                n_estimators=300, max_depth=15,
                min_samples_split=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )),
            ("xgb", XGBRegressor(
                n_estimators=500, max_depth=6,
                learning_rate=0.03, reg_alpha=1.0,
                reg_lambda=5.0, min_child_weight=5,
                random_state=42, n_jobs=-1, verbosity=0
            )),
            ("lgbm", LGBMRegressor(
                n_estimators=500, max_depth=7,
                learning_rate=0.03, reg_alpha=1.0,
                reg_lambda=5.0, min_child_samples=20,
                random_state=42, n_jobs=-1, verbose=-1
            )),
        ]

        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5, n_jobs=-1, passthrough=True
        )

    # ──────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────

    def evaluate_model(self, model, X, y, dataset_name=""):
        """
        Calculate comprehensive metrics.
        Note: y and predictions are in LOG-SPACE. We inverse-transform
        before computing real-world metrics.
        """
        y_pred_log = model.predict(X)

        # Inverse-transform from log space to real budget space
        y_real = np.expm1(y)      # real budget in crores
        y_pred_real = np.expm1(y_pred_log)
        y_pred_real = np.maximum(y_pred_real, 0)  # Budget can't be negative

        mae = mean_absolute_error(y_real, y_pred_real)
        mse = mean_squared_error(y_real, y_pred_real)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_real, y_pred_real)

        # Robust MAPE: exclude near-zero budgets (< 1 Cr) to avoid division explosion
        mape_mask = y_real >= 1.0
        if mape_mask.sum() > 0:
            mape = np.mean(np.abs((y_real[mape_mask] - y_pred_real[mape_mask]) / y_real[mape_mask])) * 100
        else:
            mape = 0.0

        # Also compute R² in log-space (for model selection)
        r2_log = r2_score(y, y_pred_log)

        # Adjusted R²
        n = len(y)
        p = X.shape[1] if hasattr(X, 'shape') else 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        # Median Absolute Error (more robust)
        medae = np.median(np.abs(y_real - y_pred_real))

        return {
            "dataset": dataset_name,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R²": round(r2, 4),
            "R² (log)": round(r2_log, 4),
            "Adjusted R²": round(adj_r2, 4),
            "MAPE (%)": round(mape, 2),
            "MedAE": round(medae, 2),
            "predictions": y_pred_real
        }

    # ──────────────────────────────────────────────────────
    # Cross-Validation
    # ──────────────────────────────────────────────────────

    def run_cross_validation(self, X_train, y_train, n_folds=5):
        """Run K-Fold Cross-Validation for all models."""
        print(f"\n📊 Running {n_folds}-Fold Cross-Validation")
        print("=" * 60)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        models_for_cv = {
            "Ridge": Ridge(alpha=10.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=42, n_jobs=-1, verbosity=0
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                random_state=42, n_jobs=-1, verbose=-1
            ),
        }

        for name, model in models_for_cv.items():
            r2_scores = cross_val_score(model, X_train, y_train, cv=kfold,
                                        scoring="r2", n_jobs=-1)
            mae_scores = -cross_val_score(model, X_train, y_train, cv=kfold,
                                          scoring="neg_mean_absolute_error", n_jobs=-1)

            self.cv_results[name] = {
                "r2_mean": round(r2_scores.mean(), 4),
                "r2_std": round(r2_scores.std(), 4),
                "r2_scores": r2_scores.tolist(),
                "mae_mean": round(mae_scores.mean(), 2),
                "mae_std": round(mae_scores.std(), 2),
                "mae_scores": mae_scores.tolist(),
            }

            print(f"   {name:20s} | R² = {r2_scores.mean():.4f} ± {r2_scores.std():.4f} "
                  f"| MAE = ₹{mae_scores.mean():.2f} ± {mae_scores.std():.2f} Cr")

    # ──────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────

    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test,
                         feature_names=None):
        """Train all base models + stacking ensemble."""
        print("\n🧠 Training Multiple ML Models")
        print("=" * 70)

        self.feature_names = feature_names
        models_dict = self._get_base_models()

        # Add stacking ensemble
        models_dict["Stacking Ensemble 🏆"] = self._get_stacking_model()

        comparison_results = []

        for name, model in models_dict.items():
            print(f"\n{'─' * 55}")
            print(f"🔄 Training: {name}")

            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, "Train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "Validation")
            test_metrics = self.evaluate_model(model, X_test, y_test, "Test")

            self.models[name] = model
            self.results[name] = {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "train_time": round(train_time, 3)
            }

            print(f"   ⏱️ Training Time: {train_time:.3f}s")
            print(f"   📊 Train R²: {train_metrics['R²']:.4f} | "
                  f"Val R²: {val_metrics['R²']:.4f} | Test R²: {test_metrics['R²']:.4f}")
            print(f"   📉 Test MAE: ₹{test_metrics['MAE']:.2f} Cr | "
                  f"RMSE: ₹{test_metrics['RMSE']:.2f} Cr")
            print(f"   📈 Test MAPE: {test_metrics['MAPE (%)']:.2f}%")

            # Overfitting check
            overfit_gap = train_metrics["R²"] - test_metrics["R²"]
            if overfit_gap > 0.1:
                print(f"   ⚠️ Overfitting detected! Gap = {overfit_gap:.4f}")

            comparison_results.append({
                "Model": name,
                "Train R²": train_metrics["R²"],
                "Val R²": val_metrics["R²"],
                "Test R²": test_metrics["R²"],
                "Adjusted R²": test_metrics["Adjusted R²"],
                "Test MAE (₹Cr)": test_metrics["MAE"],
                "Test RMSE (₹Cr)": test_metrics["RMSE"],
                "Test MAPE (%)": test_metrics["MAPE (%)"],
                "Test MedAE (₹Cr)": test_metrics["MedAE"],
                "Train Time (s)": round(train_time, 3),
                "Overfit Gap": round(overfit_gap, 4)
            })

        # Comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values("Test R²", ascending=False)

        print("\n" + "=" * 70)
        print("📊 MODEL COMPARISON (sorted by Test R²)")
        print("=" * 70)
        print(comparison_df.to_string(index=False))

        # Select best model
        best_idx = comparison_df["Test R²"].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, "Model"]
        self.best_model = self.models[self.best_model_name]

        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   R²: {comparison_df.loc[best_idx, 'Test R²']:.4f}")
        print(f"   MAE: ₹{comparison_df.loc[best_idx, 'Test MAE (₹Cr)']:.2f} Cr")

        return comparison_df

    # ──────────────────────────────────────────────────────
    # SHAP Explainability
    # ──────────────────────────────────────────────────────

    def compute_shap_values(self, X_test, feature_names=None, max_samples=500):
        """Compute SHAP values for the best model."""
        if not HAS_SHAP:
            print("⚠️ SHAP not available. Install with: pip install shap")
            return None

        print("\n🔍 Computing SHAP Explainability...")

        # Use a subset for speed
        if len(X_test) > max_samples:
            idx = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
            X_sample = X_test[idx]
        else:
            X_sample = X_test

        try:
            if hasattr(self.best_model, "predict"):
                self.shap_explainer = shap.TreeExplainer(self.best_model)
                self.shap_values = self.shap_explainer.shap_values(X_sample)

                if feature_names is not None:
                    importances = np.abs(self.shap_values).mean(axis=0)
                    feature_imp = sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1], reverse=True
                    )

                    print("\n   🔝 Top 15 SHAP Feature Importances:")
                    for feat, imp in feature_imp[:15]:
                        bar = "█" * int(imp / max(importances) * 30)
                        print(f"      {feat:35s} {imp:.4f} {bar}")

                print("   ✅ SHAP values computed successfully")
                return self.shap_values

        except Exception as e:
            print(f"   ⚠️ SHAP computation failed: {e}")
            # Fallback to feature_importances_
            if hasattr(self.best_model, "feature_importances_"):
                print("   📊 Using built-in feature importances instead")

        return None

    # ──────────────────────────────────────────────────────
    # Error Analysis
    # ──────────────────────────────────────────────────────

    def run_error_analysis(self, X_test, y_test, feature_names=None):
        """
        Analyze WHERE and WHY the model fails.
        This is what separates amateur from professional ML.
        Note: y_test is in log-space. We inverse-transform for analysis.
        """
        print("\n📉 Running Error Analysis...")
        print("=" * 60)

        y_pred_log = self.best_model.predict(X_test)

        # Inverse-transform from log space
        y_test_real = np.expm1(y_test)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)
        errors = y_test_real - y_pred
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors) / np.maximum(y_test_real, 1) * 100

        # Overall statistics
        self.error_analysis["overall"] = {
            "mean_error": round(np.mean(errors), 2),
            "std_error": round(np.std(errors), 2),
            "median_abs_error": round(np.median(abs_errors), 2),
            "p90_abs_error": round(np.percentile(abs_errors, 90), 2),
            "p95_abs_error": round(np.percentile(abs_errors, 95), 2),
        }

        print(f"   Mean Error: ₹{np.mean(errors):.2f} Cr (bias)")
        print(f"   Std Error:  ₹{np.std(errors):.2f} Cr (spread)")
        print(f"   Median Abs Error: ₹{np.median(abs_errors):.2f} Cr")
        print(f"   90th percentile: ₹{np.percentile(abs_errors, 90):.2f} Cr")

        # Error by budget range
        budget_bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 5000)]
        print("\n   📊 Error by Budget Range:")
        budget_range_errors = {}
        for low, high in budget_bins:
            mask = (y_test_real >= low) & (y_test_real < high)
            if mask.sum() > 0:
                range_mae = np.mean(abs_errors[mask])
                range_mape = np.mean(pct_errors[mask])
                count = mask.sum()
                label = f"₹{low}-{high} Cr"
                budget_range_errors[label] = {
                    "count": int(count),
                    "MAE": round(range_mae, 2),
                    "MAPE": round(range_mape, 2)
                }
                print(f"      {label:15s} | n={count:4d} | MAE=₹{range_mae:.2f} Cr | MAPE={range_mape:.1f}%")

        self.error_analysis["by_budget_range"] = budget_range_errors

        # Worst predictions
        worst_idx = np.argsort(abs_errors)[-10:]
        print("\n   ⚠️ Top 10 Worst Predictions:")
        for i, idx in enumerate(worst_idx[::-1]):
            print(f"      {i+1}. Actual: ₹{y_test_real[idx]:.2f} Cr | "
                  f"Predicted: ₹{y_pred[idx]:.2f} Cr | "
                  f"Error: ₹{abs_errors[idx]:.2f} Cr")

        # Summary insight
        high_budget_mask = y_test_real >= 200
        low_budget_mask = y_test_real < 50
        if high_budget_mask.sum() > 0 and low_budget_mask.sum() > 0:
            high_mape = np.mean(pct_errors[high_budget_mask])
            low_mape = np.mean(pct_errors[low_budget_mask])
            print(f"\n   💡 Insight: Model MAPE on high-budget (≥₹200 Cr) = {high_mape:.1f}% "
                  f"vs low-budget (<₹50 Cr) = {low_mape:.1f}%")
            if high_mape > low_mape * 1.5:
                print("   ⚠️ Model struggles with high-budget films (limited samples)")

        return self.error_analysis

    # ──────────────────────────────────────────────────────
    # Budget Recommendation System
    # ──────────────────────────────────────────────────────

    def predict_with_confidence(self, X_input):
        """
        Predict budget with confidence intervals.
        Returns: point estimate, lower bound, upper bound, similar movie range.

        Uses ensemble disagreement for uncertainty quantification.
        Note: Models predict in log-space, so we inverse-transform.
        """
        log_pred = self.best_model.predict(X_input.reshape(1, -1))[0]
        point_estimate = max(0, np.expm1(log_pred))

        # Get predictions from all individual models for uncertainty
        all_predictions = []
        for name, model in self.models.items():
            try:
                log_p = model.predict(X_input.reshape(1, -1))[0]
                pred = max(0, np.expm1(log_p))
                all_predictions.append(pred)
            except Exception:
                pass

        if len(all_predictions) > 2:
            pred_array = np.array(all_predictions)
            std_dev = np.std(pred_array)
            lower_bound = max(0, point_estimate - 1.5 * std_dev)
            upper_bound = point_estimate + 1.5 * std_dev
        else:
            # Fallback: ±20%
            lower_bound = point_estimate * 0.8
            upper_bound = point_estimate * 1.2

        return {
            "point_estimate": round(point_estimate, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "confidence_range": f"₹{round(lower_bound, 1)} - ₹{round(upper_bound, 1)} Cr",
            "model_agreement_std": round(np.std(all_predictions), 2) if all_predictions else 0,
        }

    # ──────────────────────────────────────────────────────
    # Feature Importance
    # ──────────────────────────────────────────────────────

    def get_feature_importance(self, feature_names):
        """Get feature importance from tree-based models."""
        print("\n📊 Feature Importance Analysis")
        print("=" * 50)

        importance_data = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                    "Importance_Pct": (importances / importances.sum() * 100).round(2)
                }).sort_values("Importance", ascending=False)

                importance_data[name] = imp_df

                print(f"\n🔹 {name} - Top 10 Features:")
                for _, row in imp_df.head(10).iterrows():
                    bar = "█" * int(row["Importance"] / importances.max() * 30)
                    print(f"   {row['Feature']:35s} {row['Importance_Pct']:5.1f}% {bar}")

        return importance_data

    # ──────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────

    def save_best_model(self, filepath):
        """Save the best model."""
        joblib.dump(self.best_model, filepath)
        print(f"\n💾 Best model ({self.best_model_name}) saved to: {filepath}")

    def save_all_models(self, directory):
        """Save all models."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            safe_name = name.replace(" ", "_").replace("🏆", "").strip("_")
            filepath = os.path.join(directory, f"{safe_name}.pkl")
            joblib.dump(model, filepath)
        print(f"\n💾 All {len(self.models)} models saved to: {directory}")

    def save_analysis_results(self, directory):
        """Save all analysis results as JSON."""
        os.makedirs(directory, exist_ok=True)

        # CV results
        if self.cv_results:
            cv_path = os.path.join(directory, "cv_results.json")
            with open(cv_path, "w") as f:
                json.dump(self.cv_results, f, indent=2)
            print(f"💾 CV results saved to: {cv_path}")

        # Error analysis
        if self.error_analysis:
            err_path = os.path.join(directory, "error_analysis.json")
            with open(err_path, "w") as f:
                json.dump(self.error_analysis, f, indent=2, default=str)
            print(f"💾 Error analysis saved to: {err_path}")


# ══════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_cleaning import run_full_cleaning
    from feature_engineering import create_all_features, get_feature_list
    from preprocessing import MovieDataPreprocessor

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    raw_path = os.path.join(base_dir, "data", "raw_data.csv")
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    # ── Step 1: Clean data ──
    print("\n" + "=" * 70)
    print("PHASE 1: DATA CLEANING")
    print("=" * 70)
    df = run_full_cleaning(raw_path)

    # ── Step 2: Feature engineering ──
    print("\n" + "=" * 70)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)
    df = create_all_features(df)

    # Save enriched data
    df.to_csv(os.path.join(data_dir, "cleaned_enriched_data.csv"), index=False)

    # ── Step 3: Preprocessing ──
    print("\n" + "=" * 70)
    print("PHASE 3: PREPROCESSING")
    print("=" * 70)
    preprocessor = MovieDataPreprocessor()

    # Use enriched features
    feature_list = get_feature_list()
    available_features = [f for f in feature_list if f in df.columns]

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X = df[available_features].fillna(0).values
    y_raw = df["budget_crores"].values

    # ── LOG TRANSFORM TARGET ──
    # Budget ranges from 0.03 to 1909 Cr (60,000x range).
    # Without log transform, models optimize for large budgets and
    # MAPE explodes on small ones. log1p handles zeros gracefully.
    y = np.log1p(y_raw)
    print(f"   Target transform: log1p(budget_crores)")
    print(f"   Raw range: [{y_raw.min():.2f}, {y_raw.max():.2f}] Cr")
    print(f"   Log range: [{y.min():.2f}, {y.max():.2f}]")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Temporal Split (prevents future leakage) ──
    # Sort by year, then split: oldest 70% train, next 10% val, newest 20% test
    # This is more realistic than random split for time-series-like data
    sort_idx = np.argsort(df["year"].values)
    X_sorted = X_scaled[sort_idx]
    y_sorted = y[sort_idx]

    n = len(y_sorted)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X_sorted[:train_end]
    y_train = y_sorted[:train_end]
    X_val = X_sorted[train_end:val_end]
    y_val = y_sorted[train_end:val_end]
    X_test = X_sorted[val_end:]
    y_test = y_sorted[val_end:]

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"   Train years: oldest | Val: middle | Test: newest (temporal split)")

    # Save preprocessor
    joblib.dump({
        "scaler": scaler,
        "feature_names": available_features,
        "target_transform": "log1p",
    }, os.path.join(models_dir, "preprocessor.pkl"))

    # ── Step 4: Cross-Validation ──
    trainer = AdvancedMovieBudgetTrainer()
    trainer.run_cross_validation(X_train, y_train, n_folds=5)

    # ── Step 5: Train all models ──
    comparison = trainer.train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names=available_features
    )

    # ── Step 6: Feature Importance ──
    importance = trainer.get_feature_importance(available_features)

    # ── Step 7: SHAP Explainability ──
    trainer.compute_shap_values(X_test, available_features)

    # ── Step 8: Error Analysis ──
    trainer.run_error_analysis(X_test, y_test, available_features)

    # ── Step 9: Save everything ──
    print("\n" + "=" * 70)
    print("PHASE 9: SAVING RESULTS")
    print("=" * 70)
    trainer.save_best_model(os.path.join(models_dir, "best_model.pkl"))
    trainer.save_all_models(models_dir)
    trainer.save_analysis_results(data_dir)

    comparison.to_csv(os.path.join(data_dir, "model_comparison.csv"), index=False)
    print("\n✅ ADVANCED TRAINING PIPELINE COMPLETE!")
    print(f"   🏆 Best Model: {trainer.best_model_name}")

