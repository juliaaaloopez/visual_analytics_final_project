import pandas as pd
import plotly.express as px
import streamlit as st
from itertools import combinations

from app_utils import (
    format_feature_name,
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)

df = load_products()
ing_group_df = load_ingredient_groups()
feature_info = preprocess_feature_lists(df, ing_group_df)

ingredient_cols = feature_info["ingredient_group_cols"]
highlight_cols = feature_info["highlight_cols"]
category_options = feature_info["category_options"]

st.markdown('<h1 style="color:#c51b7d;">Ingredient & Highlight Tag Insights</h1>', unsafe_allow_html=True)
st.caption("Deep dive into formulation and marketing claims behind Sephora popularity scores.")

tabs = st.tabs([
    "Intro",
    "Category Insights",
    "Ingredient Explorer",
    "Highlight Explorer",
    "Top Ingredient / Highlight Combinations",
])


def compute_combo_effects(
    data: pd.DataFrame,
    candidate_cols,
    top_limit: int = 12,
    min_support: int = 8,
):
    if data.empty or "popularity_score" not in data.columns:
        return pd.DataFrame(columns=["Combo Size", "Combination", "Products", "Avg Popularity", "Uplift vs Avg"])

    valid_cols = [col for col in candidate_cols if col in data.columns]
    if not valid_cols:
        return pd.DataFrame(columns=["Combo Size", "Combination", "Products", "Avg Popularity", "Uplift vs Avg"])

    col_counts = (
        data[valid_cols]
        .sum()
        .sort_values(ascending=False)
    )
    top_cols = col_counts[col_counts >= min_support].head(top_limit).index.tolist()
    if not top_cols:
        return pd.DataFrame(columns=["Combo Size", "Combination", "Products", "Avg Popularity", "Uplift vs Avg"])

    baseline = data["popularity_score"].mean()
    rows = []
    for size in (2, 3):
        if len(top_cols) < size:
            continue
        for combo in combinations(top_cols, size):
            mask = data[list(combo)].all(axis=1)
            support = int(mask.sum())
            if support < min_support:
                continue
            avg_pop = data.loc[mask, "popularity_score"].mean()
            uplift = avg_pop - baseline
            rows.append({
                "Combo Size": size,
                "Combination": " + ".join(format_feature_name(col) for col in combo),
                "Products": support,
                "Avg Popularity": round(float(avg_pop), 2),
                "Uplift vs Avg": round(float(uplift), 2),
            })

    if not rows:
        return pd.DataFrame(columns=["Combo Size", "Combination", "Products", "Avg Popularity", "Uplift vs Avg"])

    return (
        pd.DataFrame(rows)
        .sort_values(["Combo Size", "Avg Popularity"], ascending=[True, False])
        .reset_index(drop=True)
    )

with tabs[0]:
    st.markdown(
        """
        ## Ingredient & Highlight Tag Insights
        We originally expected ingredient families and highlight tags to dominate product popularity, but the data shows
        they're secondary to engagement metrics.  Still, ingredients and highlights play a supporting role, and this page
        explores when and where they make a noticeable difference.
        """
    )

with tabs[1]:
    st.markdown("## Category-Level Insights")
    cat_choice = st.selectbox("Select a category", options=category_options, index=1)
    cat_df = df if cat_choice == "All Categories" else df[df["primary_category"] == cat_choice]
    if cat_df.empty:
        st.warning("No records found for this category.")
    else:
        top_ing = (
            cat_df[ingredient_cols]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .rename(format_feature_name)
            .reset_index(name="Share")
            .rename(columns={"index": "Ingredient"})
        )
        ing_fig = px.bar(top_ing, x="Share", y="Ingredient", orientation="h", title="Top ingredient families",color_discrete_sequence=["#f098b0"]  )
        st.plotly_chart(ing_fig, use_container_width=True)

        pop_hist = px.histogram(
            cat_df,
            x="popularity_score",
            nbins=25,
            title="Popularity distribution within category",
            color_discrete_sequence=["#ff4778"] ,
        )
        st.plotly_chart(pop_hist, use_container_width=True)

        top_tags = (
            cat_df[highlight_cols]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .rename(format_feature_name)
            .reset_index(name="Share")
            .rename(columns={"index": "Tag"})
        )
        tag_fig = px.bar(top_tags, x="Share", y="Tag", orientation="h", title="Top highlight tags", color_discrete_sequence=["#ad284c"]  )
        st.plotly_chart(tag_fig, use_container_width=True)

        metric_cols = ["price_usd", "rating", "reviews", "loves_count"]
        metric_cols = [col for col in metric_cols if col in cat_df.columns]
        if metric_cols:
            metrics = st.columns(len(metric_cols))
            for idx, metric in enumerate(metric_cols):
                value = cat_df[metric].mean()
                formatted = f"${value:,.0f}" if metric == "price_usd" else f"{value:,.1f}" if metric == "rating" else f"{value:,.0f}"
                metrics[idx].metric(metric.replace("_", " ").title(), formatted)

with tabs[2]:
    st.markdown("## Ingredient Family Popularity Explorer")
    default_ingredients = ingredient_cols[:5]
    selected_ingredients = st.multiselect(
        "Pick ingredient families",
        options=ingredient_cols,
        default=default_ingredients,
    )
    category_filter = st.selectbox(
        "Filter by primary category",
        options=category_options,
        index=0,
        key="ingredient_category_filter",
    )
    ingredient_df = df if category_filter == "All Categories" else df[df["primary_category"] == category_filter]
    if not selected_ingredients:
        st.info("Select at least one ingredient family to see insights.")
    elif ingredient_df.empty:
        st.warning("No rows match the current category filter.")
    else:
        rows = []
        comparison_rows = []
        for col in selected_ingredients:
            if col not in ingredient_df.columns:
                continue
            has_feature = ingredient_df[ingredient_df[col] == 1]
            no_feature = ingredient_df[ingredient_df[col] == 0]
            avg_pop = has_feature["popularity_score"].mean()
            label = format_feature_name(col)
            rows.append({"Ingredient": label, "Avg Popularity": avg_pop})
            for group_name, slice_df in [("Has Feature", has_feature), ("No Feature", no_feature)]:
                if slice_df.empty:
                    continue
                comparison_rows.append(
                    {
                        "Ingredient": label,
                        "Group": group_name,
                        "Popularity Rate": slice_df.get("popularity_proxy", pd.Series(dtype=float)).mean(),
                    }
                )
        if rows:
            metric_df = pd.DataFrame(rows).dropna()
            st.dataframe(metric_df.set_index("Ingredient"))
        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows).dropna()
            comp_df["Popularity Rate"] = comp_df["Popularity Rate"] * 100
            comp_fig = px.bar(
                comp_df,
                x="Ingredient",
                y="Popularity Rate",
                color="Group",
                barmode="group",
                title="Popular vs. non-popular share",
                color_discrete_sequence=["#a20f4e", "#ff448f"]
            )
            comp_fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(comp_fig, use_container_width=True)
        else:
            st.info("Not enough data to contrast popularity rates for these ingredients.")
        with st.expander("Quick interpretation"):
            st.write(
                "Wide gaps between the paired bars (hot and dark pink) indicate that the ingredient family has a strong association with popularity in this slice."
            )

with tabs[3]:
    st.markdown("## Highlight Tag Explorer")
    selected_tags = st.multiselect(
        "Choose highlight tags",
        options=highlight_cols,
        default=highlight_cols[:5],
    )
    
    highlight_category_filter = st.selectbox(
        "Optional category filter",
        options=category_options,
        index=0,
        key="tag_category_filter",
    )
    tag_df = df if highlight_category_filter == "All Categories" else df[df["primary_category"] == highlight_category_filter]
    if not selected_tags:
        st.info("Pick at least one tag to explore.")
    elif tag_df.empty:
        st.warning("No data for this category slice.")
    else:
        freq_rows = []
        pop_rows = []
        for tag in selected_tags:
            if tag not in tag_df.columns:
                continue
            label = format_feature_name(tag)
            freq_rows.append({"Tag": label, "Frequency %": tag_df[tag].mean() * 100})
            has_tag = tag_df[tag] == 1
            pop_with = tag_df.loc[has_tag, "popularity_score"].mean()
            pop_without = tag_df.loc[~has_tag, "popularity_score"].mean()
            pop_rows.extend(
                [
                    {"Tag": label, "Group": "Has Tag", "Avg Popularity": pop_with},
                    {"Tag": label, "Group": "No Tag", "Avg Popularity": pop_without},
                ]
            )
        freq_df = pd.DataFrame(freq_rows).dropna()
        if not freq_df.empty:
            st.dataframe(freq_df.set_index("Tag"))
        pop_df = pd.DataFrame(pop_rows).dropna()
        if not pop_df.empty:
            pop_fig = px.bar(
                pop_df,
                x="Tag",
                y="Avg Popularity",
                color="Group",
                barmode="group",
                title="Popularity score with vs. without tag",
                color_discrete_sequence=["#761638", "#de1163"]
            )
            st.plotly_chart(pop_fig, use_container_width=True)
        with st.expander("Quick interpretation"):
            st.write("Wide gaps between the paired bars (dark and hot pink) suggest the tag has a meaningful link to popularity in this context.")

with tabs[4]:
    st.markdown("## Top Ingredient / Highlight Combinations")
    filter_cols = st.columns(3)
    combo_category = filter_cols[0].selectbox(
        "Category focus",
        options=category_options,
        index=0,
        key="combo_category_filter",
    )
    min_support = filter_cols[1].slider(
        "Minimum products",
        min_value=3,
        max_value=30,
        value=8,
        step=1,
        key="combo_min_support",
    )
    top_limit = filter_cols[2].slider(
        "Top features scanned",
        min_value=6,
        max_value=20,
        value=12,
        step=1,
        key="combo_top_limit",
    )

    combo_df = df if combo_category == "All Categories" else df[df["primary_category"] == combo_category]

    df_ingredient_pairs = compute_combo_effects(combo_df, ingredient_cols, top_limit=top_limit, min_support=min_support)
    df_highlight_pairs = compute_combo_effects(combo_df, highlight_cols, top_limit=top_limit, min_support=min_support)

    st.subheader("Top Ingredient Pairs & Trios")
    if df_ingredient_pairs.empty:
        st.info("No ingredient combinations cleared the minimum support or uplift threshold yet.")
    else:
        st.caption("Uplift is measured vs. the average popularity for the selected slice.")
        st.dataframe(df_ingredient_pairs, use_container_width=True)

    st.subheader("Top Highlight Tag Pairs & Trios")
    if df_highlight_pairs.empty:
        st.info("No highlight combinations cleared the minimum support or uplift threshold yet.")
    else:
        st.caption("Uplift is measured vs. the average popularity for the selected slice.")
        st.dataframe(df_highlight_pairs, use_container_width=True)

