import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    build_data_health_summary,
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
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

st.markdown('<h1 style="color:#c51b7d;"> Introduction & Overview</h1>', unsafe_allow_html=True)

st.caption("Business driven analytics connecting beauty catalog features, machine learning model, and explainable insights.")

st.header("1. Project Overview")
st.write(
    """
    This initiative blends Sephora's catalog intelligence with supervised machine learning to understand what
    differentiates promising launches. We focus on explainable signals so merchandising, marketing, and suppliers
    can act on transparent drivers that we will explain.
    """
)

st.header("2. Dataset Description")
st.markdown(
    f"""
    - **Source**: Kaggle Sephora (beauty brand/shop) products dataset enriched with feature engineering.
    - **Footprint**: {num_products:,} products across {num_brands:,} brands and {num_categories:,} primary categories.
    """
)
st.markdown(
    """
    **Feature groups**
    - **Engagement metrics**: loves_count, ratings, verified reviews volume.
    - **Price and value**: price_usd, kit sizing, price tiers.
    - **Ingredient families**: mapped groupings from parsed ingredient statements.
    - **Highlight tags**: Sephora marketing copy converted into binary product claims.
    - **Category context**: primary/secondary/tertiary taxonomy and online-only flags.
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
    The continuous **popularity_score** is a weighted combination of normalized loves, ratings, and review depth,
    giving higher influence to signals with greater variance. We convert that score into a binary proxy where
    the top 30% (`popularity_proxy = 1`) represent popular products. This top-30% cutoff gives us a practical label for
    what counts as a ‘popular’ product. It captures both how much attention a product gets and how positively people 
    react to it, which makes it a useful target for our model.”
    """
)

st.header("4. Project Motivation")
st.write(
    """
    Sephora's merchandising teams need repeatable playbooks to brief vendors, curate exclusives, and manage
    new launch investments. Our initial hypothesis is that specific ingredient families and
    marketing highlights materially shift popularity odds, so understanding those signals accelerates assortment
    and go-to-market decisions.
    """
)

st.header("5. Business Questions")
st.markdown(
    """
    - **What features really drive popularity?** Identify the characteristics that provide strong uplifts.
    - **Can ingredients and highlights explain popularity?** Make an exploration of these feature groups.
    - **How do features influence in the popularity of each category?** Study category-specific trends.
    """
)

st.header("6. What This Streamlit App Provides")
st.markdown(
    """
    - Exploratory data analysis widgets to slice the Sephora catalog.
    - Explainable machine-learning insights (feature importance, SHAP-style uplift views).
    - A prediction page to simulate new product concepts and see predicted popularity outcomes.
    """
)

st.markdown("---")
st.subheader("Interactive catalog preview")
left, right = st.columns(2)
category_choice = left.selectbox(
    "Filter by primary category or choose all",
    options=["All Categories"] + sorted(products_df["primary_category"].dropna().unique()),
    index=0,
)
brand_subset = right.multiselect(
    "Filter by specific brands or choose all",
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
    color_discrete_sequence=["#d21894"] 
)
st.plotly_chart(popularity_fig, use_container_width=True)
