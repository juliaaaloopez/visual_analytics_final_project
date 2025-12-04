import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Sephora Product Insights", layout="wide")

DATA_DIR = Path("data")
MODEL_READY_PATH = DATA_DIR / "products_final.csv"
INGREDIENT_FEATURES_PATH = DATA_DIR / "ingredient_features_selected.csv"

@st.cache_data
def load_products(path: Path = MODEL_READY_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure highlight/ingredient dummies are numeric
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
    df[binary_cols] = df[binary_cols].fillna(0).astype(int)
    return df

@st.cache_data
def load_ingredient_groups(path: Path = INGREDIENT_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["group_name", "token"])
    return pd.read_csv(path, sep=";")

@st.cache_data
def build_data_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isna().sum(),
        "unique_count": df.nunique(),
    })
    summary["dtype"] = summary["dtype"].astype(str)
    summary["missing_pct"] = (summary["missing_count"] / len(df)).round(4)
    return summary.sort_values("missing_pct", ascending=False)

def compute_feature_uplift(df: pd.DataFrame, features: list[str], target: str = "popularity_score") -> pd.DataFrame:
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
        rows.append({
            "feature": col,
            "avg_with_feature": means[1],
            "avg_without_feature": means[0],
            "uplift": means[1] - means[0],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("uplift", ascending=False)
        if rows
        else pd.DataFrame(columns=["feature", "avg_with_feature", "avg_without_feature", "uplift"])
    )

def compute_price_delta(df: pd.DataFrame, features: list[str], price_col: str = "price_usd") -> pd.DataFrame:
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
        rows.append({
            "feature": col,
            "avg_price_with": means[1],
            "avg_price_without": means[0],
            "delta": means[1] - means[0],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("delta", ascending=False)
        if rows
        else pd.DataFrame(columns=["feature", "avg_price_with", "avg_price_without", "delta"])
    )

def format_feature_name(name: str) -> str:
    return name.replace("tag_", "").replace("_", " ").title()

products_df = load_products()
ing_group_df = load_ingredient_groups()
highlight_cols = [c for c in products_df.columns if c.startswith("tag_")]
ingredient_group_cols = [
    grp for grp in ing_group_df["group_name"].unique().tolist() if grp in products_df.columns
]
# Fallback: also consider columns that start with "has_"
ingredient_group_cols.extend(
    [c for c in products_df.columns if c.startswith("has_") and c not in ingredient_group_cols]
)
ingredient_group_cols = list(dict.fromkeys(ingredient_group_cols))

st.title("Sephora Product Insights Hub")
st.caption(
    "Answer key business questions about assortment strength, pricing, and popularity drivers."
)

overview_tab, category_tab, popularity_tab, pricing_tab, benchmark_tab = st.tabs(
    [
        "🏠 Overview",
        "🛍️ Category & Sales",
        "⭐ Popularity Drivers",
        "💲 Pricing Drivers",
        "📊 Benchmarks",
    ]
)

with overview_tab:
    st.subheader("Key Portfolio Snapshot")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Products", f"{len(products_df):,}")
    kpi_cols[1].metric("Median Price", f"${products_df['price_usd'].median():.2f}")
    kpi_cols[2].metric("Median Loves", f"{products_df['loves_count'].median():,.0f}")
    kpi_cols[3].metric("Categories", products_df["primary_category"].nunique())
    st.write(
        "These KPIs reflect the cleaned dataset exported from the EDA pipeline (Section 1), ensuring consistent category and feature engineering."
    )
    with st.expander("Data Quality Overview"):
        st.dataframe(build_data_health_summary(products_df).head(25))
    st.markdown("---")
    st.subheader("Distribution Highlights")
    dist_cols = st.columns(2)
    price_fig = px.histogram(products_df, x="price_usd", nbins=40, title="Price Distribution")
    dist_cols[0].plotly_chart(price_fig, use_container_width=True)
    loves_fig = px.histogram(
        products_df,
        x="loves_count",
        nbins=40,
        title="Engagement (Loves) Distribution",
    )
    dist_cols[1].plotly_chart(loves_fig, use_container_width=True)

with category_tab:
    st.subheader("Category Coverage & Demand")
    product_id_col = "product_id" if "product_id" in products_df.columns else "product_name"
    category_summary = (
        products_df.groupby("primary_category")
        .agg(
            products=(product_id_col, "nunique"),
            avg_price=("price_usd", "mean"),
            avg_loves=("loves_count", "mean"),
            avg_reviews=("reviews", "mean"),
        )
        .sort_values("products", ascending=False)
        .round(2)
    )
    st.write(
        "The chart ranks categories by assortment size and average engagement to reveal which areas drive most traffic."
    )
    st.plotly_chart(
        px.bar(
            category_summary.reset_index().head(15),
            x="products",
            y="primary_category",
            orientation="h",
            title="Top Categories by Product Count",
            labels={"primary_category": "Category", "products": "Product Count"},
        ),
        use_container_width=True,
    )
    st.markdown("### Mean Price by Category")
    st.write(
        "Business Question: *What is the mean price by category?* Use the table to benchmark pricing ladders."
    )
    st.dataframe(category_summary[["avg_price"]].rename(columns={"avg_price": "Mean Price USD"}))
    st.markdown("### Engagement by Category")
    st.plotly_chart(
        px.scatter(
            category_summary.reset_index(),
            x="avg_price",
            y="avg_loves",
            size="products",
            color="primary_category",
            title="Average Loves vs. Mean Price",
            labels={"primary_category": "Category", "avg_price": "Mean Price", "avg_loves": "Avg Loves"},
        ),
        use_container_width=True,
    )

with popularity_tab:
    st.subheader("Popularity vs. Ingredients & Highlights")
    st.write(
        "Business Question: *Is there any relation between product popularity and ingredients/highlights?* Hover over the tables to spot boosters."
    )
    popularity_metric = st.radio(
        "Popularity metric",
        ["popularity_score", "popularity_proxy"],
        index=0,
        format_func=lambda x: "Popularity Score" if x == "popularity_score" else "Binary Proxy",
        horizontal=True,
    )
    highlight_uplift = compute_feature_uplift(products_df, highlight_cols, popularity_metric)
    ingredient_uplift = compute_feature_uplift(products_df, ingredient_group_cols, popularity_metric)
    st.markdown("#### Highlight Impact")
    st.dataframe(
        highlight_uplift.head(15).assign(
            feature=lambda df_: df_["feature"].apply(format_feature_name)
        )
    )
    st.markdown("#### Ingredient Group Impact")
    st.dataframe(ingredient_uplift.head(15))
    st.caption("Positive uplift indicates higher popularity when the feature is present.")

with pricing_tab:
    st.subheader("Price Drivers by Feature & Category")
    st.write(
        "Business Question: *Which ingredients/highlights make products more expensive by category?* Select a category to benchmark price deltas."
    )
    price_metric_cols = st.columns(2)
    highlight_price = compute_price_delta(products_df, highlight_cols)
    ingredient_price = compute_price_delta(products_df, ingredient_group_cols)
    price_metric_cols[0].markdown("#### Highlight Price Delta (All Categories)")
    price_metric_cols[0].dataframe(
        highlight_price.head(15).assign(
            feature=lambda df_: df_["feature"].apply(format_feature_name)
        )
    )
    price_metric_cols[1].markdown("#### Ingredient Price Delta (All Categories)")
    price_metric_cols[1].dataframe(ingredient_price.head(15))
    st.markdown("### Category-Level Price Diagnostics")
    selected_category = st.selectbox(
        "Choose category",
        options=sorted(products_df["primary_category"].dropna().unique()),
    )
    filtered_df = products_df[products_df["primary_category"] == selected_category]
    if filtered_df.empty:
        st.info("No products in the selected category.")
    else:
        top_highlights = compute_price_delta(filtered_df, highlight_cols).head(10)
        top_ingredients = compute_price_delta(filtered_df, ingredient_group_cols).head(10)
        cols = st.columns(2)
        cols[0].write("Top highlight premiums in category")
        cols[0].dataframe(
            top_highlights.assign(feature=lambda df_: df_["feature"].apply(format_feature_name))
        )
        cols[1].write("Top ingredient premiums in category")
        cols[1].dataframe(top_ingredients)

with benchmark_tab:
    st.subheader("Supporting Benchmarks & Data Context")
    st.write("Use these visuals to explain broader distribution and quality trends.")
    brand_summary = (
        products_df["brand_name"]
        .value_counts()
        .reset_index(name="product_count")
        .rename(columns={"index": "brand_name"})
    )
    bench_cols = st.columns(2)
    bench_cols[0].plotly_chart(
        px.bar(
            brand_summary.head(10),
            x="brand_name",
            y="product_count",
            title="Top 10 Brands by Product Count",
        ),
        use_container_width=True,
    )
    bench_cols[1].plotly_chart(
        px.box(
            products_df,
            x="primary_category",
            y="price_usd",
            points="suspectedoutliers",
            title="Price spread by category",
        ),
        use_container_width=True,
    )
    st.markdown("### Raw Data Preview")
    st.dataframe(products_df.head(50))
    st.caption("Export-ready data originates from the EDA pipeline cleaning functions in Section 1.")

