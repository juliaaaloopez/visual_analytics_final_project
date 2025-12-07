import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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


def get_artifact(name: str):
    return st.session_state.get(name, globals().get(name))

st.title("Ingredient & Highlight Tag Insights")
st.caption("Deep dive into formulation and marketing claims behind Sephora popularity scores.")

tabs = st.tabs([
    "Intro",
    "Ingredient Explorer",
    "Highlight Explorer",
    "Category Insights",
    "SHAP by Category",
    "Takeaways",
])

with tabs[0]:
    st.markdown(
        """
        ## Ingredient & Highlight Tag Insights
        We originally expected ingredient families and highlight tags to dominate popularity, but the data shows
        they're secondary to engagement metrics. Still, formulation and copy cues nudge performance, so this page
        explores where they matter most.
        """
    )

with tabs[1]:
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
            )
            comp_fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(comp_fig, use_container_width=True)
        else:
            st.info("Not enough data to contrast popularity rates for these ingredients.")
        with st.expander("Quick interpretation"):
            st.write(
                "Look for big gaps between the purple bars—those indicate formulations that really tilt popularity."
            )

with tabs[2]:
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
            )
            st.plotly_chart(pop_fig, use_container_width=True)
        with st.expander("How to read this"):
            st.write("Wide gaps between the paired bars mean the tag changes average popularity in that slice.")

with tabs[3]:
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
        ing_fig = px.bar(top_ing, x="Share", y="Ingredient", orientation="h", title="Top ingredient families")
        st.plotly_chart(ing_fig, use_container_width=True)

        pop_hist = px.histogram(
            cat_df,
            x="popularity_score",
            nbins=25,
            title="Popularity distribution within category",
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
        tag_fig = px.bar(top_tags, x="Share", y="Tag", orientation="h", title="Top highlight tags")
        st.plotly_chart(tag_fig, use_container_width=True)

        metric_cols = ["price_usd", "rating", "reviews", "loves_count"]
        metric_cols = [col for col in metric_cols if col in cat_df.columns]
        if metric_cols:
            metrics = st.columns(len(metric_cols))
            for idx, metric in enumerate(metric_cols):
                value = cat_df[metric].mean()
                formatted = f"${value:,.0f}" if metric == "price_usd" else f"{value:,.1f}" if metric == "rating" else f"{value:,.0f}"
                metrics[idx].metric(metric.replace("_", " ").title(), formatted)

with tabs[4]:
    st.markdown("## SHAP Explainability by Category")
    shap_category = st.selectbox("Choose category for SHAP view", options=category_options, index=1)
    min_samples = st.slider("Minimum samples required", 20, 200, 50, 10)

    shap_data = None
    feature_matrix = None
    feature_names = get_artifact("feature_names")
    shap_source = get_artifact("shap_values")
    X_transformed = get_artifact("X_test_transformed")

    if isinstance(shap_source, dict):
        shap_data = shap_source.get(shap_category)
    else:
        shap_data = shap_source

    if isinstance(X_transformed, dict):
        feature_matrix = X_transformed.get(shap_category)
    else:
        feature_matrix = X_transformed

    if (
        shap_data is None
        or feature_matrix is None
        or feature_names is None
        or len(feature_matrix) < min_samples
    ):
        st.warning("Not enough samples for reliable SHAP interpretation.")
    else:
        shap_array = np.array(shap_data)
        mean_abs = np.mean(np.abs(shap_array), axis=0)
        top_idx = np.argsort(mean_abs)[-5:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        shap_df = pd.DataFrame({
            "Feature": [format_feature_name(feat) for feat in top_features],
            "Mean |SHAP|": mean_abs[top_idx],
        })
        shap_fig = px.bar(
            shap_df,
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            title=f"Top SHAP drivers • {shap_category}",
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.write(
            "These are the signals the category-level model leans on most—larger bars mean bigger sway on predicted"
            " popularity."
        )

with tabs[5]:
    st.markdown("## Takeaways")
    st.markdown(
        """
        - Ingredient impact shifts by category: skincare reacts to humectants while makeup leans on finish claims.
        - Highlight tags rarely beat engagement metrics, but they provide the narrative glue for high-scoring launches.
        - The original hypothesis was partly right—formulations matter—but only when paired with strong social proof.
        """
    )
