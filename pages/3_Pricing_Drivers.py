import streamlit as st

from app_utils import (
    compute_price_uplift,
    format_feature_name,
    format_percentage,
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)

products_df = load_products()
ing_group_df = load_ingredient_groups()
feature_info = preprocess_feature_lists(products_df, ing_group_df)

highlight_cols = feature_info["highlight_cols"]
ingredient_group_cols = feature_info["ingredient_group_cols"]
category_options = feature_info["category_options"]

st.title("Pricing Drivers")
st.caption("Identify which highlights and ingredient groups command price premiums by category.")

category_choice = st.selectbox(
    "Choose category to inspect",
    options=category_options,
    index=0,
)
top_n = st.slider("Top N rows", 5, 25, 10)
keyword = st.text_input("Optional feature keyword filter").strip().lower()

filtered_df = (
    products_df
    if category_choice == "All Categories"
    else products_df[products_df["primary_category"] == category_choice]
)

if filtered_df.empty:
    st.warning("No products available for the selected category.")
else:
    def prepare_table(uplift_df, feature_label):
        if keyword:
            uplift_df = uplift_df[uplift_df["feature"].str.contains(keyword, case=False, na=False)]
        if uplift_df.empty:
            return None
        return (
            uplift_df.assign(
                feature=lambda df_: df_["feature"].apply(format_feature_name),
                avg_price_with=lambda df_: df_["avg_price_with"].round(2),
                avg_price_without=lambda df_: df_["avg_price_without"].round(2),
                uplift_pct=lambda df_: df_["uplift_pct"].apply(format_percentage),
            )
            .rename(
                columns={
                    "feature": feature_label,
                    "avg_price_with": "Avg Price • Has Feature",
                    "avg_price_without": "Avg Price • No Feature",
                    "uplift_pct": "Uplift %",
                }
            )
            .head(top_n)
        )

    st.subheader("Highlight price uplift")
    highlight_price = compute_price_uplift(filtered_df, highlight_cols)
    highlight_table = prepare_table(highlight_price, "Highlight Feature")
    if highlight_table is not None:
        st.dataframe(highlight_table)
    else:
        st.info("No highlight features with measurable price uplift in this slice.")

    st.subheader("Ingredient group price uplift")
    ingredient_price = compute_price_uplift(filtered_df, ingredient_group_cols)
    ingredient_table = prepare_table(ingredient_price, "Ingredient Group")
    if ingredient_table is not None:
        st.dataframe(ingredient_table)
    else:
        st.info("No ingredient groups with measurable price uplift in this slice.")

st.caption("Positive uplift indicates a higher average price when the feature is present.")
