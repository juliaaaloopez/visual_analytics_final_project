import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    build_data_health_summary,
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)

st.set_page_config(
    page_title="Sephora Product Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

products_df = load_products()
ing_group_df = load_ingredient_groups()
feature_info = preprocess_feature_lists(products_df, ing_group_df)

num_products = len(products_df)
num_brands = products_df["brand_name"].nunique()
num_categories = products_df["primary_category"].nunique()
median_price = products_df["price_usd"].median()
rating_series = products_df["rating"] if "rating" in products_df.columns else pd.Series(dtype=float)
reviews_series = products_df["reviews"] if "reviews" in products_df.columns else pd.Series(dtype=float)
avg_rating = rating_series.mean()
avg_reviews = reviews_series.mean()
popular_share = (
    products_df["popularity_proxy"].mean()
    if "popularity_proxy" in products_df.columns
    else None
)
popularity_cutoff = (
    products_df["popularity_score"].quantile(0.70)
    if "popularity_score" in products_df.columns
    else None
)
popularity_cutoff_text = f"{popularity_cutoff:,.2f}" if popularity_cutoff is not None else "N/A"

stats_rows = [
    {"Metric": "Median Price (USD)", "Value": f"${median_price:,.2f}"},
    {"Metric": "Average Rating", "Value": f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A"},
    {"Metric": "Average Reviews", "Value": f"{avg_reviews:,.0f}" if not pd.isna(avg_reviews) else "N/A"},
]
if popular_share is not None:
    stats_rows.append(
        {
            "Metric": "Share of Popular SKUs",
            "Value": f"{popular_share*100:,.1f}%",
        }
    )

st.title("Introduction & Overview")
st.caption("Executive-facing analytics connecting catalog features, machine learning, and explainable insights.")

st.header("1. Project Overview")
st.write(
    """
    This initiative blends Sephora's catalog intelligence with supervised machine learning to understand what
    differentiates breakout launches. We focus on explainable signals—so merchandising, marketing, and suppliers
    can act on transparent drivers rather than opaque scores.
    """
)

st.header("2. Dataset Description")
st.markdown(
    f"""
    - **Source**: Kaggle Sephora products dataset enriched with in-house feature engineering.
    - **Footprint**: {num_products:,} products across {num_brands:,} brands and {num_categories:,} primary categories.
    """
)
st.markdown(
    """
    **Feature groups**
    - Engagement metrics: loves_count, rating signal, verified reviews volume.
    - Price and value: price_usd, kit sizing, price tiers.
    - Ingredient families: mapped groupings from parsed ingredient statements.
    - Highlight tags: Sephora marketing copy converted into binary product claims.
    - Category context: primary/secondary/tertiary taxonomy and online-only flags.
    """
)
st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

with st.expander("Data quality snapshot"):
    dq_summary = build_data_health_summary(products_df).drop(
        index=feature_info["binary_columns"], errors="ignore"
    )
    st.dataframe(dq_summary[["dtype", "unique_count"]].head(25))

st.header("3. Defining Product Popularity")
st.write(
    """
    The continuous **popularity_score** is a weighted ensemble of normalized loves, ratings, and review depth,
    giving higher influence to signals with greater variance. We convert that score into a binary proxy where
    the top 30% (`popularity_proxy = 1`) represent popular SKUs. This percentile cut (≈{popularity_cutoff_text} score)
    captures both velocity and quality sentiment, making it a pragmatic dependent variable for explainable ML.
    """
)

st.header("4. Project Motivation")
st.write(
    """
    Sephora's merchandising teams need repeatable playbooks to brief vendors, curate exclusives, and manage
    launch investments before hard sales data accrues. Our hypothesis is that specific ingredient families and
    marketing highlights materially shift popularity odds, so understanding those signals accelerates assortment
    and go-to-market decisions.
    """
)

st.header("5. Business Questions")
st.markdown(
    """
    - **What drives popularity?** Identify the quantitative levers with the strongest uplift.
    - **Can ingredients or tags explain popularity?** Traceable explainability for formulation and marketing teams.
    - **How do categories differ?** Surface category-level nuances that warrant bespoke strategies.
    - Secondary: Which formulation traits (clean, vegan, fragrance-free, etc.) act as leading indicators?
    """
)

st.header("6. What This Streamlit App Provides")
st.markdown(
    """
    - Exploratory data analysis widgets to slice the catalog on demand.
    - Explainable machine-learning insights (feature importance, SHAP-style uplift views).
    - Category and pricing benchmarks for cross-portfolio comparisons.
    - A lightweight prediction sandbox to pressure-test new product concepts.
    """
)

st.markdown("---")
st.subheader("Interactive catalog probe")
left, right = st.columns(2)
category_choice = left.selectbox(
    "Filter by primary category",
    options=["All Categories"] + sorted(products_df["primary_category"].dropna().unique()),
    index=0,
)
brand_subset = right.multiselect(
    "Highlight specific brands",
    options=sorted(products_df["brand_name"].unique()),
    default=[],
)
price_min, price_max = st.slider(
    "Price range (USD)",
    float(products_df["price_usd"].min()),
    float(products_df["price_usd"].max()),
    (float(products_df["price_usd"].min()), float(products_df["price_usd"].max())),
)

filtered_df = products_df.copy()
if category_choice != "All Categories":
    filtered_df = filtered_df[filtered_df["primary_category"] == category_choice]
if brand_subset:
    filtered_df = filtered_df[filtered_df["brand_name"].isin(brand_subset)]
filtered_df = filtered_df[
    (filtered_df["price_usd"] >= price_min) & (filtered_df["price_usd"] <= price_max)
]

preview_cols = [
    col
    for col in [
        "product_name",
        "brand_name",
        "primary_category",
        "price_usd",
        "popularity_score",
        "popularity_proxy",
    ]
    if col in filtered_df.columns
]
st.dataframe(filtered_df[preview_cols].head(50))

st.markdown("### Popularity score distribution")
popularity_fig = px.histogram(
    products_df,
    x="popularity_score",
    nbins=40,
    title="Distribution of Popularity Score",
)
st.plotly_chart(popularity_fig, use_container_width=True)
