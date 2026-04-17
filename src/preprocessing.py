"""
🧹 Data Preprocessing Module
==============================
Handles missing values, encoding, normalization, and data splitting
for the Indian Movie Budget Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import warnings
warnings.filterwarnings("ignore")


class MovieDataPreprocessor:
    """
    Comprehensive preprocessor for Indian movie dataset.
    Handles cleaning, encoding, scaling, and feature selection.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = []
        self.categorical_columns = [
            "industry", "language", "genre", "certification", "verdict"
        ]
        self.numerical_columns = [
            "year", "release_month", "director_success_rate",
            "actor_popularity_score", "num_cast_members", "runtime_minutes",
            "is_sequel", "num_songs", "vfx_level", "num_screens",
            "international_release", "ott_release"
        ]
        self.target_column = "budget_crores"
    
    def load_data(self, filepath):
        """Load raw CSV data."""
        print("📂 Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"   ✅ Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def explore_data(self, df):
        """Print basic EDA statistics."""
        print("\n🔍 Exploratory Data Analysis")
        print("=" * 50)
        
        print(f"\n📊 Shape: {df.shape}")
        print(f"\n🔢 Data Types:\n{df.dtypes}")
        
        print(f"\n❌ Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   None! Dataset is clean.")
        else:
            print(missing[missing > 0])
        
        print(f"\n📈 Numerical Statistics:")
        print(df[self.numerical_columns + [self.target_column]].describe().round(2))
        
        print(f"\n🏷️ Categorical Distributions:")
        for col in self.categorical_columns:
            if col in df.columns:
                print(f"\n   {col}:")
                print(f"   {df[col].value_counts().to_dict()}")
        
        return df
    
    def clean_data(self, df):
        """Handle missing values and outliers."""
        print("\n🧹 Cleaning Data...")
        
        initial_len = len(df)
        
        # Drop duplicates
        df = df.drop_duplicates()
        print(f"   Removed {initial_len - len(df)} duplicate rows")
        
        # Handle missing numerical values with median
        for col in self.numerical_columns:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   Filled {col} missing values with median: {median_val:.2f}")
        
        # Handle missing categorical values with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"   Filled {col} missing values with mode: {mode_val}")
        
        # Remove extreme outliers in budget (> 99.5th percentile)
        budget_cap = df[self.target_column].quantile(0.995)
        outliers = len(df[df[self.target_column] > budget_cap])
        df = df[df[self.target_column] <= budget_cap]
        print(f"   Removed {outliers} budget outlier rows (> ₹{budget_cap:.2f} Cr)")
        
        # Ensure runtime is reasonable
        df = df[(df["runtime_minutes"] >= 60) & (df["runtime_minutes"] <= 240)]
        
        print(f"   ✅ Cleaned dataset: {len(df)} records")
        return df
    
    def encode_features(self, df, fit=True):
        """Encode categorical variables."""
        print("\n🔄 Encoding Categorical Features...")
        
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns and col != "verdict":
                if fit:
                    le = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df_encoded[f"{col}_encoded"] = le.transform(df_encoded[col])
                print(f"   ✅ Encoded {col}: {len(le.classes_)} categories")
        
        return df_encoded
    
    def engineer_features(self, df):
        """Create derived features for better prediction."""
        print("\n⚙️ Engineering Features...")
        
        # Budget per minute
        # (will use this as a target-adjacent feature for analysis only)
        
        # Decade feature
        df["decade"] = (df["year"] // 10) * 10
        print("   ✅ Added: decade")
        
        # Season feature (festival seasons matter for Indian cinema)
        def get_season(month):
            if month in [1, 4, 10, 11]:
                return 3  # Festival / Peak season
            elif month in [3, 5, 8, 12]:
                return 2  # Moderate season
            else:
                return 1  # Off season
        
        df["release_season"] = df["release_month"].apply(get_season)
        print("   ✅ Added: release_season")
        
        # Star power metric (combination of actor popularity and director success)
        df["star_power"] = (
            df["actor_popularity_score"] * 0.6 + 
            df["director_success_rate"] * 10 * 0.4
        )
        print("   ✅ Added: star_power")
        
        # Production scale (VFX + screens + international)
        df["production_scale"] = (
            df["vfx_level"] * 0.4 + 
            (df["num_screens"] / df["num_screens"].max()) * 10 * 0.3 +
            df["international_release"] * 3 * 0.3
        )
        print("   ✅ Added: production_scale")
        
        # Industry growth factor
        industry_growth = {
            "Bollywood": 0.08, "Tollywood": 0.12, "Kollywood": 0.10,
            "Sandalwood": 0.09, "Mollywood": 0.11
        }
        df["industry_growth_factor"] = df.apply(
            lambda r: (1 + industry_growth.get(r["industry"], 0.08)) ** (r["year"] - 2000),
            axis=1
        )
        print("   ✅ Added: industry_growth_factor")
        
        # High budget genre flag
        expensive_genres = ["Sci-Fi", "Historical", "War", "Fantasy", "Mythology", "Action"]
        df["is_expensive_genre"] = df["genre"].isin(expensive_genres).astype(int)
        print("   ✅ Added: is_expensive_genre")
        
        return df
    
    def prepare_features(self, df, fit=True):
        """Prepare final feature matrix and target vector."""
        print("\n📋 Preparing Feature Matrix...")
        
        feature_cols = self.numerical_columns + [
            "industry_encoded", "language_encoded", "genre_encoded",
            "certification_encoded", "decade", "release_season",
            "star_power", "production_scale", "industry_growth_factor",
            "is_expensive_genre"
        ]
        
        # Keep only available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        self.feature_columns = available_cols
        
        X = df[available_cols].values
        y = df[self.target_column].values
        
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        print(f"   ✅ Features: {len(available_cols)} columns")
        print(f"   ✅ Samples: {len(X)} rows")
        
        return X, y, available_cols
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets."""
        print(f"\n✂️ Splitting Data (Test: {test_size}, Val: {val_size})...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        print(f"   ✅ Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath):
        """Save preprocessing objects for inference."""
        preprocessor_data = {
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns
        }
        joblib.dump(preprocessor_data, filepath)
        print(f"💾 Preprocessor saved to: {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessing objects for inference."""
        data = joblib.load(filepath)
        self.label_encoders = data["label_encoders"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.categorical_columns = data["categorical_columns"]
        self.numerical_columns = data["numerical_columns"]
        print(f"📂 Preprocessor loaded from: {filepath}")
    
    def run_pipeline(self, raw_data_path, save_dir=None):
        """Execute the complete preprocessing pipeline."""
        print("🚀 Running Complete Preprocessing Pipeline")
        print("=" * 60)
        
        # 1. Load
        df = self.load_data(raw_data_path)
        
        # 2. Explore
        df = self.explore_data(df)
        
        # 3. Clean
        df = self.clean_data(df)
        
        # 4. Engineer features
        df = self.engineer_features(df)
        
        # 5. Encode
        df = self.encode_features(df, fit=True)
        
        # 6. Save cleaned data
        if save_dir:
            cleaned_path = os.path.join(save_dir, "cleaned_data.csv")
            df.to_csv(cleaned_path, index=False)
            print(f"\n💾 Cleaned data saved to: {cleaned_path}")
        
        # 7. Prepare features
        X, y, feature_names = self.prepare_features(df, fit=True)
        
        # 8. Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 9. Save preprocessor
        if save_dir:
            self.save_preprocessor(os.path.join(save_dir, "..", "models", "preprocessor.pkl"))
        
        print("\n" + "=" * 60)
        print("✅ Preprocessing Pipeline Complete!")
        
        return {
            "df": df,
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "feature_names": feature_names
        }


if __name__ == "__main__":
    preprocessor = MovieDataPreprocessor()
    
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")
    save_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    results = preprocessor.run_pipeline(raw_path, save_dir)
