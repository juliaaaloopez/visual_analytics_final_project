import plotly.express as px
import streamlit as st

from app_utils import (
    get_price_segments,
    get_product_identifier,
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)

products_df = load_products()
ing_group_df = load_ingredient_groups()
preprocess_feature_lists(products_df, ing_group_df)  # ensures caches warm even if unused here

st.title("Benchmarks & Context")
st.caption("Supporting visuals to explain broader distribution, pricing tiers, and feature mix.")

st.subheader("Brand footprint and premium positioning")
top_brands = st.slider("Show top N brands", 5, 20, 10)
brand_summary = (
    products_df["brand_name"].value_counts().reset_index(name="product_count").rename(columns={"index": "brand_name"})
)
brand_price = (
    products_df.groupby("brand_name")["price_usd"].mean().sort_values(ascending=False).head(top_brands).reset_index()
)
bench_cols = st.columns(2)
bench_cols[0].plotly_chart(
    px.bar(
        brand_summary.head(top_brands),
        x="brand_name",
        y="product_count",
        title="Brands by product count",
    ),
    use_container_width=True,
)
bench_cols[1].plotly_chart(
    px.bar(
        brand_price,
        x="brand_name",
        y="price_usd",
        title="Brands by average price",
        labels={"price_usd": "Average price (USD)"},
    ),
    use_container_width=True,
)

st.markdown("---")
st.subheader("Price tiers vs engagement")
price_segments = get_price_segments(products_df)
id_col = get_product_identifier(products_df)
price_segment_summary = (
    products_df.assign(price_segment=price_segments)
    .groupby("price_segment")
    .agg(avg_loves=("loves_count", "mean"), products=(id_col, "count"))
    .reset_index()
)
segment_fig = px.scatter(
    price_segment_summary,
    x="price_segment",
    y="avg_loves",
    size="products",
    title="Engagement by price segment",
    labels={"avg_loves": "Average loves", "price_segment": "Price segment"},
)
st.plotly_chart(segment_fig, use_container_width=True)

st.markdown("---")
st.subheader("Price spread by category")
price_categories = st.multiselect(
    "Filter categories",
    options=sorted(products_df["primary_category"].dropna().unique()),
    default=sorted(products_df["primary_category"].dropna().unique())[:6],
)
if price_categories:
    price_spread_df = products_df[products_df["primary_category"].isin(price_categories)]
else:
    price_spread_df = products_df
price_box = px.box(
    price_spread_df,
    x="primary_category",
    y="price_usd",
    title="Price spread by category",
    points="suspectedoutliers",
)
st.plotly_chart(price_box, use_container_width=True)

st.markdown("---")
st.subheader("Feature volume vs popularity proxy")
feature_metric = st.selectbox(
    "Choose feature count",
    options={"n_highlights": "Highlight count", "n_ingredients": "Ingredient count"},
    format_func=lambda key: {"n_highlights": "Highlight count", "n_ingredients": "Ingredient count"}[key],
)
feature_box = px.box(
    products_df,
    x="popularity_proxy",
    y=feature_metric,
    color="popularity_proxy",
    title=f"{feature_metric.replace('_', ' ').title()} vs popularity proxy",
    labels={
        "popularity_proxy": "Popularity proxy",
        feature_metric: feature_metric.replace("_", " ").title(),
    },
)
st.plotly_chart(feature_box, use_container_width=True)
