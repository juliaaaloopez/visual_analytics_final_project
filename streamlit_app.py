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
        "unique_count": df.nunique(),
        "memory_bytes": df.memory_usage(deep=True),
    })
    summary["dtype"] = summary["dtype"].astype(str)
    summary["memory_bytes"] = summary["memory_bytes"].astype(int)
    return summary.sort_values("memory_bytes", ascending=False)

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
    return (
        pd.DataFrame(rows)
        .sort_values("uplift_pct", ascending=False)
        if rows
        else pd.DataFrame(columns=["feature", "avg_with_feature", "avg_without_feature", "uplift_pct"])
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
    return (
        pd.DataFrame(rows)
        .sort_values("uplift_pct", ascending=False)
        if rows
        else pd.DataFrame(columns=["feature", "avg_price_with", "avg_price_without", "uplift_pct"])
    )

def format_feature_name(name: str) -> str:
    return name.replace("tag_", "").replace("_", " ").title()

def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"

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
binary_columns = [
    col
    for col in products_df.columns
    if products_df[col].dropna().isin([0, 1]).all()
]
category_options = ["All Categories"] + sorted(
    products_df["primary_category"].dropna().unique().tolist()
)

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
    st.markdown("### Raw Data Preview")
    st.dataframe(products_df.head(50))
    with st.expander("Data Quality Overview"):
        dq_cols = ["dtype", "unique_count"]
        dq_summary = build_data_health_summary(products_df).drop(index=binary_columns, errors="ignore")
        st.dataframe(dq_summary[dq_cols].head(25))
    st.markdown("---")
    st.subheader("Distribution Highlights")
    dist_rows = [st.columns(2), st.columns(2)]
    price_fig = px.histogram(products_df, x="price_usd", nbins=40, title="Price Distribution")
    dist_rows[0][0].plotly_chart(price_fig, use_container_width=True)
    log_reviews = np.log1p(products_df["reviews"].clip(lower=0))
    log_reviews_fig = px.histogram(
        x=log_reviews,
        nbins=40,
        title="Log Reviews Distribution",
        labels={"x": "log(1 + reviews)"},
    )
    dist_rows[0][1].plotly_chart(log_reviews_fig, use_container_width=True)
    log_loves = np.log1p(products_df["loves_count"].clip(lower=0))
    log_loves_fig = px.histogram(
        x=log_loves,
        nbins=40,
        title="Log Loves Count Distribution",
        labels={"x": "log(1 + loves_count)"},
    )
    dist_rows[1][0].plotly_chart(log_loves_fig, use_container_width=True)
    ratings_fig = px.histogram(
        products_df,
        x="reviews",
        nbins=40,
        title="Number of Ratings Distribution",
        labels={"reviews": "Number of Ratings"},
    )
    dist_rows[1][1].plotly_chart(ratings_fig, use_container_width=True)

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
            avg_rating=("rating", "mean"),
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
    metric_cols = st.columns(2)
    avg_reviews_fig = px.bar(
        category_summary.reset_index().sort_values("avg_reviews", ascending=False),
        x="primary_category",
        y="avg_reviews",
        title="Average Reviews by Category",
        labels={"primary_category": "Category", "avg_reviews": "Avg Reviews"},
    )
    metric_cols[0].plotly_chart(avg_reviews_fig, use_container_width=True)
    avg_rating_fig = px.bar(
        category_summary.reset_index().sort_values("avg_rating", ascending=False),
        x="primary_category",
        y="avg_rating",
        title="Average Rating by Category",
        labels={"primary_category": "Category", "avg_rating": "Avg Rating"},
    )
    metric_cols[1].plotly_chart(avg_rating_fig, use_container_width=True)

with popularity_tab:
    st.subheader("Popularity vs. Ingredients & Highlights")
    st.write(
        "Business Question: *Is there any relation between product popularity and ingredients/highlights?* Hover over the tables to spot boosters."
    )
    category_choice = st.selectbox(
        "Select category to inspect",
        options=category_options,
        index=0,
        key="popularity_category_selector",
    )
    popularity_filtered_df = (
        products_df
        if category_choice == "All Categories"
        else products_df[products_df["primary_category"] == category_choice]
    )
    highlight_uplift = compute_feature_uplift(popularity_filtered_df, highlight_cols, "popularity_score")
    ingredient_uplift = compute_feature_uplift(popularity_filtered_df, ingredient_group_cols, "popularity_score")

    st.markdown("#### Highlight Impact")
    if highlight_uplift.empty:
        st.info("Not enough highlight variation in this slice to compute uplift.")
    else:
        highlight_table = (
            highlight_uplift.assign(
                feature=lambda df_: df_["feature"].apply(format_feature_name),
                avg_with_feature=lambda df_: df_["avg_with_feature"].round(2),
                avg_without_feature=lambda df_: df_["avg_without_feature"].round(2),
                uplift_pct=lambda df_: df_["uplift_pct"].apply(format_percentage),
            )
            .rename(
                columns={
                    "feature": "Highlight Feature",
                    "avg_with_feature": "Avg Popularity • Has Feature",
                    "avg_without_feature": "Avg Popularity • No Feature",
                    "uplift_pct": "Uplift %",
                }
            )
            .head(10)
        )
        st.dataframe(highlight_table)

    st.markdown("#### Ingredient Group Impact")
    if ingredient_uplift.empty:
        st.info("Not enough ingredient signals in this slice to compute uplift.")
    else:
        ingredient_table = (
            ingredient_uplift.assign(
                feature=lambda df_: df_["feature"].apply(format_feature_name),
                avg_with_feature=lambda df_: df_["avg_with_feature"].round(2),
                avg_without_feature=lambda df_: df_["avg_without_feature"].round(2),
                uplift_pct=lambda df_: df_["uplift_pct"].apply(format_percentage),
            )
            .rename(
                columns={
                    "feature": "Ingredient Group",
                    "avg_with_feature": "Avg Popularity • Has Feature",
                    "avg_without_feature": "Avg Popularity • No Feature",
                    "uplift_pct": "Uplift %",
                }
            )
            .head(10)
        )
        st.dataframe(ingredient_table)
    st.caption("Positive uplift indicates higher popularity when the feature is present.")

with pricing_tab:
    st.subheader("Price Drivers by Feature & Category")
    st.write(
        "Business Question: *Which ingredients/highlights make products more expensive by category?* Select a category to benchmark price uplifts."
    )
    price_category_choice = st.selectbox(
        "Select category to inspect",
        options=category_options,
        index=0,
        key="pricing_category_selector",
    )
    pricing_filtered_df = (
        products_df
        if price_category_choice == "All Categories"
        else products_df[products_df["primary_category"] == price_category_choice]
    )
    highlight_price = compute_price_delta(pricing_filtered_df, highlight_cols)
    ingredient_price = compute_price_delta(pricing_filtered_df, ingredient_group_cols)

    st.markdown("#### Highlight Price Uplift")
    if highlight_price.empty:
        st.info("Not enough highlight variation in this slice to compute uplift.")
    else:
        highlight_price_table = (
            highlight_price.assign(
                feature=lambda df_: df_["feature"].apply(format_feature_name),
                avg_price_with=lambda df_: df_["avg_price_with"].round(2),
                avg_price_without=lambda df_: df_["avg_price_without"].round(2),
                uplift_pct=lambda df_: df_["uplift_pct"].apply(format_percentage),
            )
            .rename(
                columns={
                    "feature": "Highlight Feature",
                    "avg_price_with": "Avg Price • Has Feature",
                    "avg_price_without": "Avg Price • No Feature",
                    "uplift_pct": "Uplift %",
                }
            )
            .head(10)
        )
        st.dataframe(highlight_price_table)

    st.markdown("#### Ingredient Price Uplift")
    if ingredient_price.empty:
        st.info("Not enough ingredient signals in this slice to compute uplift.")
    else:
        ingredient_price_table = (
            ingredient_price.assign(
                feature=lambda df_: df_["feature"].apply(format_feature_name),
                avg_price_with=lambda df_: df_["avg_price_with"].round(2),
                avg_price_without=lambda df_: df_["avg_price_without"].round(2),
                uplift_pct=lambda df_: df_["uplift_pct"].apply(format_percentage),
            )
            .rename(
                columns={
                    "feature": "Ingredient Group",
                    "avg_price_with": "Avg Price • Has Feature",
                    "avg_price_without": "Avg Price • No Feature",
                    "uplift_pct": "Uplift %",
                }
            )
            .head(10)
        )
        st.dataframe(ingredient_price_table)

with benchmark_tab:
    st.subheader("Supporting Benchmarks & Data Context")
    st.write("Use these visuals to explain broader distribution and quality trends.")
    brand_summary = (
        products_df["brand_name"]
        .value_counts()
        .reset_index(name="product_count")
        .rename(columns={"index": "brand_name"})
    )
    brand_price = (
        products_df.groupby("brand_name")["price_usd"].mean().sort_values(ascending=False).head(10).reset_index()
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
        px.bar(
            brand_price,
            x="brand_name",
            y="price_usd",
            title="Top 10 Brands by Average Price",
            labels={"price_usd": "Average Price (USD)"},
        ),
        use_container_width=True,
    )

    st.markdown("### Price Tiers vs. Engagement")
    price_bins = pd.qcut(products_df["price_usd"], q=[0, 0.25, 0.5, 0.75, 1], duplicates="drop")
    price_labels = ["Budget", "Mass", "Prestige", "Luxury"][: len(price_bins.cat.categories)]
    price_segments = price_bins.cat.rename_categories(price_labels)
    id_col_benchmark = "product_id" if "product_id" in products_df.columns else "product_name"
    price_segment_summary = (
        products_df.assign(price_segment=price_segments)
        .groupby("price_segment")
        .agg(avg_loves=("loves_count", "mean"), products=(id_col_benchmark, "count"))
        .reset_index()
    )
    st.plotly_chart(
        px.scatter(
            price_segment_summary,
            x="price_segment",
            y="avg_loves",
            size="products",
            title="Engagement by Price Segment",
            labels={"avg_loves": "Average Loves", "price_segment": "Price Segment"},
        ),
        use_container_width=True,
    )

    st.markdown("### Price Spread by Category")
    st.plotly_chart(
        px.box(
            products_df,
            x="primary_category",
            y="price_usd",
            points="suspectedoutliers",
            title="Price Spread by Category",
        ),
        use_container_width=True,
    )

    st.markdown("### Feature Volume vs. Popularity")
    feature_cols = st.columns(2)
    feature_cols[0].plotly_chart(
        px.box(
            products_df,
            x="popularity_proxy",
            y="n_highlights",
            color="popularity_proxy",
            title="Highlight Count vs Popularity Proxy",
            labels={"popularity_proxy": "Popularity Proxy", "n_highlights": "Number of Highlights"},
        ),
        use_container_width=True,
    )
    feature_cols[1].plotly_chart(
        px.box(
            products_df,
            x="popularity_proxy",
            y="n_ingredients",
            color="popularity_proxy",
            title="Ingredient Count vs Popularity Proxy",
            labels={"popularity_proxy": "Popularity Proxy", "n_ingredients": "Number of Ingredients"},
        ),
        use_container_width=True,
    )

