import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)


def get_session_object(name: str):
    return st.session_state.get(name)


def prepare_model_input(row_df: pd.DataFrame, model) -> pd.DataFrame:
    if model is None:
        return row_df
    feature_names_in = getattr(model, "feature_names_in_", None)
    if feature_names_in is None:
        return row_df
    missing = [col for col in feature_names_in if col not in row_df.columns]
    if missing:
        st.warning(
            "Some required model features are missing from the current selection. Prediction may be inaccurate."
        )
        return row_df
    return row_df[feature_names_in]


def transform_for_shap(model, model_input: pd.DataFrame):
    if model is None:
        return None
    transformer = None
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        transformer = model.named_steps["preprocessor"]
    elif hasattr(model, "steps"):
        # fall back to first step if it looks like a transformer
        first_step = model.steps[0][1]
        if hasattr(first_step, "transform"):
            transformer = first_step
    if transformer is not None:
        transformed = transformer.transform(model_input)
    else:
        transformed = model_input.values
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed


def compute_shap_values(explainer, transformed_row, feature_names) -> pd.DataFrame:
    if explainer is None or transformed_row is None:
        return pd.DataFrame()
    try:
        shap_values = explainer.shap_values(transformed_row)
    except AttributeError:
        shap_values = explainer(transformed_row).values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    sample = shap_values[0]
    feature_count = len(sample)
    name_list = (
        list(feature_names)
        if feature_names is not None and len(feature_names) == feature_count
        else [f"Feature {i}" for i in range(feature_count)]
    )
    df = pd.DataFrame({"Feature": name_list, "SHAP Value": sample})
    return df.assign(Impact=lambda d: d["SHAP Value"].abs()).sort_values("Impact", ascending=False)


def build_prediction_summary(pred_label, popularity_probability):
    is_popular = bool(pred_label)
    verdict = "Popular" if is_popular else "Not Popular"
    tone = st.success if is_popular else st.warning
    tone(f"Model verdict: **{verdict}** based on popularity proxy")
    if popularity_probability is not None:
        st.metric("Predicted probability (popular)", f"{popularity_probability:.2%}")


def main():
    df = load_products()
    ing_group_df = load_ingredient_groups()
    feature_info = preprocess_feature_lists(df, ing_group_df)
    popularity_model = get_session_object("popularity_model")
    explainer = get_session_object("explainer")
    feature_names = get_session_object("feature_names")

    highlight_cols = feature_info["highlight_cols"]
    ingredient_cols = feature_info["ingredient_group_cols"]

    st.markdown('<h1 style="color:#c51b7d;">Product Prediction</h1>', unsafe_allow_html=True)

    st.caption("Experiment with different product characteristics, predict if it will be popular and inspect drivers.")

    tabs = st.tabs(["Predict Product", "Conclusions"])

    with tabs[0]:
        st.markdown("### Predict a New Product Popularity")
        brand_options = sorted(df["brand_name"].dropna().unique())
        template_row = df.head(1).copy()
        for col in highlight_cols + ingredient_cols:
            if col in template_row.columns:
                template_row.loc[:, col] = 0

        top_highlights = (
            df[highlight_cols].mean().sort_values(ascending=False).head(20).index.tolist()
            if highlight_cols
            else []
        )
        top_ingredients = (
            df[ingredient_cols].mean().sort_values(ascending=False).head(20).index.tolist()
            if ingredient_cols
            else []
        )

        with st.form(key="prediction_form"):
            col_price, col_brand = st.columns(2)
            price_input = col_price.number_input(
                "Price (USD)",
                min_value=0.0,
                max_value=float(df["price_usd"].max()),
                value=float(df["price_usd"].median()),
                step=1.0,
            )
            brand_input = col_brand.selectbox("Brand", options=brand_options)

            flag_cols = st.columns(4)
            limited_flag = flag_cols[0].checkbox("Limited edition", value=False)
            new_flag = flag_cols[1].checkbox("New launch", value=True, help="Sets the `new` flag to 1")
            online_flag = flag_cols[2].checkbox("Online only", value=False)
            exclusive_flag = flag_cols[3].checkbox("Sephora exclusive", value=False)

            child_count_input = st.number_input(
                "Child count",
                min_value=0,
                value=0,
                step=1,
            )

            highlight_selection = st.multiselect(
                "Key highlight tags",
                options=top_highlights,
                default=[],
                help="Toggle the tags you want this concept SKU to carry.",
            )
            ingredient_selection = st.multiselect(
                "Key ingredient families",
                options=top_ingredients,
                default=[],
                help="Pick the big ingredient stories you want to emphasize.",
            )

            submitted = st.form_submit_button("Predict Popularity! ")

        if submitted:
            if popularity_model is None:
                st.error("Model artifacts missing. Please ensure the bundle is loaded into session state.")
            else:
                custom_row = template_row.copy()
                custom_row.loc[:, "price_usd"] = price_input
                if "rating" in custom_row.columns:
                    custom_row.loc[:, "rating"] = float(df["rating"].median())
                if "reviews" in custom_row.columns:
                    custom_row.loc[:, "reviews"] = int(df["reviews"].median())
                if "loves_count" in custom_row.columns:
                    custom_row.loc[:, "loves_count"] = int(df["loves_count"].median())
                custom_row.loc[:, "brand_name"] = brand_input
                if "limited_edition" in custom_row.columns:
                    custom_row.loc[:, "limited_edition"] = int(limited_flag)
                if "new" in custom_row.columns:
                    custom_row.loc[:, "new"] = int(new_flag)
                if "online_only" in custom_row.columns:
                    custom_row.loc[:, "online_only"] = int(online_flag)
                if "sephora_exclusive" in custom_row.columns:
                    custom_row.loc[:, "sephora_exclusive"] = int(exclusive_flag)
                if "child_count" in custom_row.columns:
                    custom_row.loc[:, "child_count"] = child_count_input

                if highlight_cols:
                    custom_row.loc[:, highlight_cols] = 0
                    for col in highlight_selection:
                        if col in custom_row.columns:
                            custom_row.loc[:, col] = 1
                if ingredient_cols:
                    custom_row.loc[:, ingredient_cols] = 0
                    for col in ingredient_selection:
                        if col in custom_row.columns:
                            custom_row.loc[:, col] = 1

                model_input = prepare_model_input(custom_row, popularity_model)
                prediction_raw = popularity_model.predict(model_input)[0]
                popularity_probability = None
                if hasattr(popularity_model, "predict_proba"):
                    popularity_probability = float(popularity_model.predict_proba(model_input)[0][-1])
                build_prediction_summary(prediction_raw, popularity_probability)

                transformed_row = transform_for_shap(popularity_model, model_input)
                shap_df = compute_shap_values(explainer, transformed_row, feature_names)

                st.markdown("### SHAP explanation for this prediction")
                if shap_df.empty:
                    st.info("SHAP artifacts unavailable. Ensure the explainer and feature names were loaded.")
                else:
                    top_shap = shap_df.head(12)
                    shap_fig = px.bar(
                        top_shap.sort_values("SHAP Value"),
                        x="SHAP Value",
                        y="Feature",
                        orientation="h",
                        title="Top feature contributions",
                        color="SHAP Value",
                        color_continuous_scale="RdPu",
                    )
                    shap_fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(shap_fig, use_container_width=True)
                    st.caption("Positive SHAP pushes the score up; negative pulls it down. Magnitude shows influence.")

    with tabs[1]:
        st.markdown("### Conclusions")
        st.markdown(
            """
            📉 **Our initial hypothesis didn’t fully hold.**
            We expected ingredient families and highlight tags to be major popularity drivers, but SHAP shows they play a secondary role compared to engagement metrics (reviews, loves, rating). They still help the model, just not as much as we originally thought.

            👀 **Popularity behaves like a visibility + social-proof problem.**
            Products succeed when they accumulate strong customer interaction signals. High review volume, strong ratings, and repeated engagement are the main levers influencing predicted popularity.

            🔗 **Ingredients and highlights matter most in combinations.**
            Individual ingredient families rarely shift outcomes, but pairs and trios (e.g., Humectants + Antioxidants + Mineral Sunscreen) consistently show uplift. Customers respond to coherent benefit stories, not isolated claims.

            📊 **Popularity varies meaningfully by category.**
            Makeup and Mini Size products outperform other groups, likely due to faster trend cycles and lower commitment barriers. Other categories, like Fragrance or Bath & Body, follow different dynamics, reinforcing category-specific popularity behavior.

            💲 **Price shows a nonlinear effect.**
            Both high-end and budget items can perform well when paired with the right engagement and messaging signals. Success isn’t about being “cheap vs. premium,” but about positioning.

            🏷️ **Operational and merchandising signals matter.**
            Features such as online-only, new, and child_count (product variants like colors) appear among the strongest SHAP contributors. This suggests popularity is influenced not just by product content but also by how and where the product is launched.

            🧠 **Overall, the model’s insights align with real consumer behavior.**
            Customers amplify products with strong social proof, clear positioning, and relevant benefit combinations. Ingredients help, but engagement drives.
            """
        )


if __name__ == "__main__":
    main()
