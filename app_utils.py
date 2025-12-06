from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
import streamlit as st

DATA_DIR = Path("data")
MODEL_READY_PATH = DATA_DIR / "products_final.csv"
INGREDIENT_FEATURES_PATH = DATA_DIR / "ingredient_features_selected.csv"

if "shap_values" not in st.session_state:
    model_bundle = joblib.load("models/popularity_bundle.pkl")
    st.session_state["popularity_model"] = model_bundle["model"]
    st.session_state["explainer"] = model_bundle["explainer"]
    st.session_state["shap_values"] = model_bundle["shap_values"]
    st.session_state["feature_names"] = model_bundle["feature_names"]
    st.session_state["X_test_transformed"] = model_bundle["X_test_transformed"]


@st.cache_data
def load_products(path: Path = MODEL_READY_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
    if binary_cols:
        df[binary_cols] = df[binary_cols].fillna(0).astype(int)
    return df


@st.cache_data
def load_ingredient_groups(path: Path = INGREDIENT_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["group_name", "token"])
    return pd.read_csv(path, sep=";")


def preprocess_feature_lists(products_df: pd.DataFrame, ing_group_df: pd.DataFrame) -> Dict[str, List[str]]:
    highlight_cols = [c for c in products_df.columns if c.startswith("tag_")]
    ingredient_group_cols = [
        grp for grp in ing_group_df["group_name"].unique().tolist() if grp in products_df.columns
    ]
    ingredient_group_cols.extend(
        [c for c in products_df.columns if c.startswith("has_") and c not in ingredient_group_cols]
    )
    ingredient_group_cols = list(dict.fromkeys(ingredient_group_cols))
    binary_columns = [
        col for col in products_df.columns if products_df[col].dropna().isin([0, 1]).all()
    ]
    category_options = ["All Categories"] + sorted(
        products_df["primary_category"].dropna().unique().tolist()
    )
    return {
        "highlight_cols": highlight_cols,
        "ingredient_group_cols": ingredient_group_cols,
        "binary_columns": binary_columns,
        "category_options": category_options,
    }


def build_data_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_count": df.nunique(),
        "memory_bytes": df.memory_usage(deep=True),
    })
    return summary.sort_values("memory_bytes", ascending=False)


def compute_feature_uplift(
    df: pd.DataFrame,
    features: List[str],
    target: str = "popularity_score",
) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in df.columns:
            continue
        col_series = df[col].fillna(0)
        if col_series.nunique() <= 1:
            continue
        means = df.groupby(col_series)[target].mean()
        if 0 not in means.index or 1 not in means.index:
            continue
        avg_without = means[0]
        avg_with = means[1]
        if pd.isna(avg_without) or avg_without == 0:
            continue
        uplift_pct = ((avg_with - avg_without) / avg_without) * 100
        rows.append({
            "feature": col,
            "avg_with_feature": avg_with,
            "avg_without_feature": avg_without,
            "uplift_pct": uplift_pct,
        })
    if not rows:
        return pd.DataFrame(columns=["feature", "avg_with_feature", "avg_without_feature", "uplift_pct"])
    return pd.DataFrame(rows).sort_values("uplift_pct", ascending=False)


def compute_price_uplift(
    df: pd.DataFrame,
    features: List[str],
    price_col: str = "price_usd",
) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in df.columns:
            continue
        col_series = df[col].fillna(0)
        if col_series.nunique() <= 1:
            continue
        means = df.groupby(col_series)[price_col].mean()
        if 0 not in means.index or 1 not in means.index:
            continue
        avg_without = means[0]
        avg_with = means[1]
        if pd.isna(avg_without) or avg_without == 0:
            continue
        uplift_pct = ((avg_with - avg_without) / avg_without) * 100
        rows.append({
            "feature": col,
            "avg_price_with": avg_with,
            "avg_price_without": avg_without,
            "uplift_pct": uplift_pct,
        })
    if not rows:
        return pd.DataFrame(columns=["feature", "avg_price_with", "avg_price_without", "uplift_pct"])
    return pd.DataFrame(rows).sort_values("uplift_pct", ascending=False)


def format_feature_name(name: str) -> str:
    return name.replace("tag_", "").replace("_", " ").title()


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def get_product_identifier(df: pd.DataFrame) -> str:
    return "product_id" if "product_id" in df.columns else "product_name"


def get_price_segments(df: pd.DataFrame) -> pd.Series:
    price_bins = pd.qcut(df["price_usd"], q=[0, 0.25, 0.5, 0.75, 1], duplicates="drop")
    labels = ["Budget", "Mass", "Prestige", "Luxury"][: len(price_bins.cat.categories)]
    return price_bins.cat.rename_categories(labels)
