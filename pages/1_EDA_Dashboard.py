import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    get_price_segments,
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)

df = load_products().copy()
ing_group_df = load_ingredient_groups()
feature_info = preprocess_feature_lists(df, ing_group_df)

highlight_cols = feature_info["highlight_cols"]
ingredient_cols = feature_info["ingredient_group_cols"]
category_options = feature_info["category_options"]

if "brand_name" not in df.columns:
    df["brand_name"] = "Unknown"
brand_options = ["All Brands"] + sorted(df["brand_name"].dropna().unique().tolist())

if "price_usd" in df.columns:
    df["price_segment"] = get_price_segments(df)
else:
    df["price_segment"] = "Unknown"

st.title("EDA Dashboard")
st.caption("Hands-on playground to poke at Sephora's popularity signals.")

st.markdown("### Global filters")
filter_cols = st.columns(2)
category_filter = filter_cols[0].selectbox(
    "Primary category",
    options=category_options,
    index=0,
    key="eda_category_filter",
)
brand_filter = filter_cols[1].selectbox(
    "Brand",
    options=brand_options,
    index=0,
    key="eda_brand_filter",
)

brand_filtered_df = df.copy()
if brand_filter != "All Brands":
    brand_filtered_df = brand_filtered_df[brand_filtered_df["brand_name"] == brand_filter]

filtered_df = brand_filtered_df.copy()
if category_filter != "All Categories":
    filtered_df = filtered_df[filtered_df["primary_category"] == category_filter]

if filtered_df.empty:
    st.warning("No records match the current category/brand combo. Try loosening the filters.")

def format_feature_label(name: str) -> str:
    return name.replace("tag_", "").replace("has_", "").replace("_", " ").title()

def render_popularity_section(data: pd.DataFrame) -> None:
    st.subheader("1. Popularity Overview")
    if data.empty or "popularity_score" not in data.columns:
        st.info("Need popularity_score values to render this section.")
        return
    bin_count = st.slider("Histogram bins", min_value=10, max_value=80, value=40, step=5, key="pop_bins")
    pop_hist = px.histogram(
        data,
        x="popularity_score",
        nbins=bin_count,
        title="Popularity score distribution",
        color_discrete_sequence=["#7b5cd6"],
    )
    st.plotly_chart(pop_hist, use_container_width=True)

    cat_bar = data.groupby("primary_category")["popularity_score"].mean().dropna().sort_values(ascending=False)
    if not cat_bar.empty:
        cat_fig = px.bar(
            cat_bar.reset_index().rename(columns={"popularity_score": "avg_popularity"}),
            x="primary_category",
            y="avg_popularity",
            title="Average popularity by category",
        )
        st.plotly_chart(cat_fig, use_container_width=True)

    brand_bar = (
        data.groupby("brand_name")["popularity_score"].mean().dropna().sort_values(ascending=False).head(20)
    )
    if not brand_bar.empty:
        brand_fig = px.bar(
            brand_bar.reset_index().rename(columns={"popularity_score": "avg_popularity"}),
            x="brand_name",
            y="avg_popularity",
            title="Top brands by avg popularity",
        )
        brand_fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(brand_fig, use_container_width=True)

def render_engagement_section(data: pd.DataFrame, brand_scope: pd.DataFrame) -> None:
    st.subheader("2. Engagement Metrics")
    if data.empty:
        st.info("No rows available for this slice.")
        return
    if "price_usd" in brand_scope.columns and not brand_scope["price_usd"].dropna().empty:
        min_price = float(brand_scope["price_usd"].min())
        max_price = float(brand_scope["price_usd"].max())
        price_min, price_max = st.slider(
            "Price range (USD)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            key="price_slider",
        )
        price_filtered_df = data[
            (data["price_usd"] >= price_min) & (data["price_usd"] <= price_max)
        ]
    else:
        st.info("Price column unavailable, so the price slider is hidden.")
        price_filtered_df = data

    color_choices = []
    if "primary_category" in data.columns:
        color_choices.append("Primary Category")
    if "popularity_proxy" in data.columns:
        color_choices.append("Popularity Proxy")
    color_by = st.radio(
        "Color scatter by",
        options=color_choices or ["None"],
        horizontal=True,
        key="scatter_color",
    )
    color_field = None
    if color_by == "Primary Category":
        color_field = "primary_category"
    elif color_by == "Popularity Proxy":
        color_field = "popularity_proxy"

    if not price_filtered_df.empty and {"rating", "reviews"}.issubset(price_filtered_df.columns):
        scatter_df = price_filtered_df.dropna(subset=["rating", "reviews"])
        if not scatter_df.empty:
            scatter_fig = px.scatter(
                scatter_df,
                x="reviews",
                y="rating",
                color=color_field,
                size="price_usd" if "price_usd" in scatter_df.columns else None,
                hover_data=[col for col in ["product_name", "brand_name"] if col in scatter_df.columns],
                title="Rating vs. Reviews",
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.info("Need both rating and reviews columns to plot the scatter chart.")

    if "loves_count" in price_filtered_df.columns:
        love_bins = st.slider("Loves count bins", min_value=10, max_value=80, value=30, step=5, key="love_bins")
        loves_fig = px.histogram(
            price_filtered_df,
            x="loves_count",
            nbins=love_bins,
            title="Loves count distribution",
        )
        st.plotly_chart(loves_fig, use_container_width=True)

    if {"rating", "price_segment"}.issubset(price_filtered_df.columns):
        violin_fig = px.box(
            price_filtered_df.dropna(subset=["rating"]),
            x="price_segment",
            y="rating",
            color="price_segment",
            title="Rating distribution by price segment",
        )
        st.plotly_chart(violin_fig, use_container_width=True)

def render_correlation_section(data: pd.DataFrame) -> None:
    st.subheader("3. Price / Rating / Reviews Relationship")
    metric_cols = [col for col in ["price_usd", "rating", "reviews", "loves_count"] if col in data.columns]
    if len(metric_cols) < 2 or data.empty:
        st.info("Need at least two of price, rating, reviews, loves_count to compute correlations.")
        return
    corr_matrix = data[metric_cols].corr()
    st.dataframe(corr_matrix.style.format("{:.2f}"), use_container_width=True)
    heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Purples",
        title="Correlation heatmap",
    )
    st.plotly_chart(heatmap, use_container_width=True)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    stacked = upper.stack()
    top_pair = stacked.reindex(stacked.abs().sort_values(ascending=False).index).head(1)
    with st.expander("What these correlations say"):
        if not top_pair.empty:
            pair = top_pair.index[0]
            value = top_pair.values[0]
            st.write(
                f"Strongest swing right now: **{pair[0]} vs {pair[1]}** at {value:.2f}."
            )
        else:
            st.write("Not enough variance to call out correlations yet.")

def render_ingredient_section(data: pd.DataFrame) -> None:
    st.subheader("4. Ingredient & Highlight Tag Exploration")
    if data.empty:
        st.info("No rows available for this slice.")
        return
    default_ingredients = ingredient_cols[:5]
    selected_ingredients = st.multiselect(
        "Ingredient families", options=ingredient_cols, default=default_ingredients, key="ingredient_multi"
    )
    if selected_ingredients:
        ing_counts, ing_popularity = [], []
        for col in selected_ingredients:
            if col not in data.columns:
                continue
            total = int(data[col].sum())
            share = (total / len(data)) * 100 if len(data) else 0
            ing_counts.append({"Ingredient": format_feature_label(col), "Products": total, "Share %": share})
            avg_pop = data.loc[data[col] == 1, "popularity_score"].mean()
            ing_popularity.append({"Ingredient": format_feature_label(col), "Avg Popularity": avg_pop})
        if ing_counts:
            st.dataframe(pd.DataFrame(ing_counts).set_index("Ingredient"))
        if ing_popularity:
            pop_fig = px.bar(
                pd.DataFrame(ing_popularity).dropna(),
                x="Ingredient",
                y="Avg Popularity",
                title="Average popularity when ingredient shows up",
            )
            st.plotly_chart(pop_fig, use_container_width=True)
    else:
        st.info("Pick one or more ingredient families to see counts and lift.")

    selected_tag = st.selectbox(
        "Highlight tag focus",
        options=highlight_cols if highlight_cols else ["No tags available"],
        key="tag_focus",
    )
    if selected_tag in data.columns:
        freq = data[selected_tag].mean() * 100 if len(data) else 0
        st.metric("Tag frequency in current slice", f"{freq:.1f}%")
        if highlight_cols:
            top_tags = (
                data[highlight_cols]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()
            )
            if top_tags and "primary_category" in data.columns:
                tag_long = (
                    data.groupby("primary_category")[top_tags]
                    .mean()
                    .reset_index()
                    .melt(id_vars="primary_category", value_name="share", var_name="tag")
                )
                tag_fig = px.bar(
                    tag_long,
                    x="primary_category",
                    y="share",
                    color="tag",
                    barmode="group",
                    title="Tag frequency by category",
                )
                st.plotly_chart(tag_fig, use_container_width=True)
    else:
        st.info("No highlight tags available to profile.")

def render_category_section(data: pd.DataFrame, brand_scope: pd.DataFrame) -> None:
    st.subheader("5. Category Deep Dive")
    dive_categories = sorted(brand_scope["primary_category"].dropna().unique().tolist())
    if not dive_categories:
        st.info("No categories left after filtering.")
        return
    deep_dive_choice = st.selectbox("Pick a category", options=dive_categories, key="deep_dive_category")
    cat_df = brand_scope[brand_scope["primary_category"] == deep_dive_choice]
    if cat_df.empty:
        st.warning("No rows in this category for the current brand filter.")
        return
    if "popularity_score" in cat_df.columns:
        cat_pop_fig = px.histogram(
            cat_df,
            x="popularity_score",
            nbins=30,
            title=f"Popularity distribution • {deep_dive_choice}",
            color_discrete_sequence=["#f25f5c"],
        )
        st.plotly_chart(cat_pop_fig, use_container_width=True)

    def summarize_binary(columns, top_n=8):
        rows = []
        for col in columns:
            if col not in cat_df.columns:
                continue
            value = cat_df[col].sum()
            rows.append({"feature": format_feature_label(col), "value": value})
        return (
            pd.DataFrame(rows)
            .sort_values("value", ascending=False)
            .head(top_n)
        )

    ing_summary = summarize_binary(ingredient_cols)
    tag_summary = summarize_binary(highlight_cols)
    ing_col, tag_col = st.columns(2)
    if not ing_summary.empty:
        ing_fig = px.bar(
            ing_summary,
            x="value",
            y="feature",
            orientation="h",
            title="Top ingredient families",
            labels={"value": "Count"},
        )
        ing_col.plotly_chart(ing_fig, use_container_width=True)
    else:
        ing_col.info("No ingredient columns to summarize.")
    if not tag_summary.empty:
        tag_fig = px.bar(
            tag_summary,
            x="value",
            y="feature",
            orientation="h",
            title="Top highlight tags",
            labels={"value": "Count"},
        )
        tag_col.plotly_chart(tag_fig, use_container_width=True)
    else:
        tag_col.info("No highlight columns to summarize.")

    metric_candidates = [col for col in ["price_usd", "rating", "loves_count"] if col in cat_df.columns]
    if metric_candidates:
        kpi_cols = st.columns(len(metric_candidates))
        for idx, col_name in enumerate(metric_candidates):
            value = cat_df[col_name].mean()
            if col_name == "price_usd":
                formatted = f"${value:,.0f}"
            elif col_name == "rating":
                formatted = f"{value:.2f}"
            else:
                formatted = f"{value:,.0f}"
            kpi_cols[idx].metric(col_name.replace("_", " ").title(), formatted)

sections = [
    ("1. Popularity Overview", "pop", render_popularity_section),
    ("2. Engagement Metrics", "engagement", render_engagement_section),
    ("3. Price / Rating / Reviews", "correlation", render_correlation_section),
    ("4. Ingredient & Highlight Tags", "ingredients", render_ingredient_section),
    ("5. Category Deep Dive", "category", render_category_section),
]

if "eda_active_section" not in st.session_state:
    st.session_state["eda_active_section"] = sections[0][1]

button_cols = st.columns(len(sections))
for idx, (label, key, _) in enumerate(sections):
    if button_cols[idx].button(label, key=f"section_btn_{key}"):
        st.session_state["eda_active_section"] = key

active_section = st.session_state["eda_active_section"]
for label, key, renderer in sections:
    if key == active_section:
        if key == "engagement":
            renderer(filtered_df, brand_filtered_df)
        elif key == "category":
            renderer(filtered_df, brand_filtered_df)
        else:
            renderer(filtered_df)
        break

st.markdown("---")
if filtered_df.empty:
    st.write("Reset the filters to see the EDA insights again.")

