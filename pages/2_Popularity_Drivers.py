import numpy as np
import plotly.express as px
import streamlit as st

from app_utils import (
    load_ingredient_groups,
    load_products,
    preprocess_feature_lists,
)

import matplotlib.pyplot as plt
import pandas as pd
import shap

LOAD_KEYS = [
    "explainer",
    "shap_values",                      # global SHAP
    "X_test_transformed",               # global X-test
    "feature_names",
    "shap_values_by_category",          # NEW
    "X_test_transformed_by_category",   # NEW
]

for key in LOAD_KEYS:
    if key in st.session_state:
        globals()[key] = st.session_state[key]

st.markdown('<h1 style="color:#c51b7d;">What Drives Popularity?</h1>', unsafe_allow_html=True)
st.caption("SHAP-powered peek into how our gradient boosting model scores Sephora launches.")

# Resolve globally provided artifacts
shap_values_obj = globals().get("shap_values")
explainer = globals().get("explainer")
X_test_transformed = globals().get("X_test_transformed")
feature_names_global = globals().get("feature_names")

if isinstance(X_test_transformed, pd.DataFrame):
    feature_frame = X_test_transformed.copy()
elif X_test_transformed is not None and feature_names_global is not None:
    feature_frame = pd.DataFrame(X_test_transformed, columns=feature_names_global)
else:
    feature_frame = None

if shap_values_obj is None:
    shap_matrix = None
elif hasattr(shap_values_obj, "values"):
    shap_matrix = shap_values_obj.values
else:
    shap_matrix = np.array(shap_values_obj)

feature_names = (
    list(feature_frame.columns)
    if feature_frame is not None
    else (list(feature_names_global) if feature_names_global is not None else [])
)

st.markdown(
    """
    ### Intro
    This page breaks down **global feature importance** using SHAP for the Gradient Boosting model we trained to
    predict product popularity. SHAP (SHapley Additive exPlanations) lets us peek into the model's logic without
    getting lost in the math, so we can explain to the business why a product gets a high or low score.
    """
)

st.markdown("---")
st.markdown("## Feature Importance (Top SHAP Features)")
top_n = st.slider("Show top N features", min_value=5, max_value=30, value=10, step=1)

if shap_matrix is None or not feature_names:
    st.info("SHAP values or feature names are missing, so the importance chart can't render yet.")
else:
    mean_abs_importance = np.mean(np.abs(shap_matrix), axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs_importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    bar_fig = px.bar(
        importance_df[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {len(importance_df)} SHAP features",
        labels={"importance": "Mean |SHAP|", "feature": "Feature"},
        color="importance",
        color_continuous_scale="RdPu",
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    with st.expander("What are we looking at?", expanded=False):
        st.write(
            "Each bar shows how much a feature moves the popularity prediction on average."      
            " Bigger bars = bigger impact."
            " Think of it as the model's power ranking of signals."
        )

st.markdown("---")
st.markdown("## SHAP Summary Plot")
show_top_only = st.checkbox("Only show the top N features from above", value=False)

if shap_matrix is None or feature_frame is None:
    st.info("Need both SHAP values and the feature matrix to draw the beeswarm plot.")
else:
    if show_top_only:
        top_features = importance_df["feature"].tolist() if "importance_df" in locals() else feature_names[:top_n]
        selected_features = [feat for feat in top_features if feat in feature_frame.columns]
    else:
        selected_features = feature_frame.columns.tolist()

    if not selected_features:
        st.info("No features available for the beeswarm selection.")
    else:
        indices = [feature_frame.columns.get_loc(feat) for feat in selected_features]
        shap_subset = shap_matrix[:, indices]
        feature_subset = feature_frame[selected_features]
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_subset, feature_subset, show=False)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            "Smaller dots hug the center when impact is tiny; further from zero means the feature is really pushing"
            " predictions. Colors show feature value (pink = high,  blue = low)."
        )

st.markdown("---")
st.markdown("## Interactive Explanation")
feature_choice = st.selectbox(
    "Pick a feature to inspect",
    options=feature_names if feature_names else ["No features available"],
)

if (
    feature_choice
    and feature_choice in feature_names
    and shap_matrix is not None
    and feature_frame is not None
):
    idx = feature_names.index(feature_choice)
    shap_values_feature = shap_matrix[:, idx]
    feature_values = feature_frame.iloc[:, idx]
    scatter_df = pd.DataFrame({
        "Feature Value": feature_values,
        "SHAP Value": shap_values_feature,
    })
    scatter_fig = px.scatter(
        scatter_df,
        x="Feature Value",
        y="SHAP Value",
        color="Feature Value",
        color_continuous_scale="Magma",
        title=f"SHAP dependence • {feature_choice}",
    )
    scatter_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.write(
        "Each dot is a product from the test set. Left/right shows how the feature nudged the prediction down/up,"
        " and the color hints at interactions."
    )
else:
    st.info("Need SHAP values, feature data, and a valid feature selection to show the dependence plot.")

shap_values_by_category = st.session_state.get("shap_values_by_category")
X_test_by_category = st.session_state.get("X_test_transformed_by_category")
safe_feature_names = st.session_state.get("feature_names", [])

st.markdown("---")
st.markdown("## SHAP by Category")

if not isinstance(shap_values_by_category, dict) or not isinstance(X_test_by_category, dict):
    st.info("Per-category SHAP values and feature matrices were not provided. Please make sure your model bundle includes shap_values_by_category and X_test_transformed_by_category.")
else:
    category_options = sorted(shap_values_by_category.keys())
    chosen_category = st.selectbox(
        "Select category for SHAP explanation",
        options=category_options,
        key="shap_category_dropdown",
    )

    shap_subset = shap_values_by_category.get(chosen_category)
    X_subset = X_test_by_category.get(chosen_category)

    
    MIN_SAMPLES = 40

    if shap_subset is None or X_subset is None:
        st.warning("Category SHAP data missing. Did you save it in your model bundle?")
    elif len(X_subset) < MIN_SAMPLES:
        st.warning(f"Not enough samples for reliable SHAP in category '{chosen_category}'. Need at least {MIN_SAMPLES}.")
    else:
        st.subheader(f"Top 10 SHAP Features • {chosen_category}")

        shap_array = np.array(shap_subset)
        X_cat_df = pd.DataFrame(X_subset, columns=safe_feature_names)

        
        mean_abs = np.mean(np.abs(shap_array), axis=0)

       
        top_idx = np.argsort(mean_abs)[-10:][::-1]
        top_features = [safe_feature_names[i] for i in top_idx]

        shap_df = pd.DataFrame({
            "Feature": top_features,
            "Mean |SHAP|": mean_abs[top_idx]
        }).sort_values("Mean |SHAP|")

        import plotly.express as px
        fig = px.bar(
            shap_df,
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            title=f"SHAP Feature Importance for {chosen_category}",
            color="Mean |SHAP|",
            color_continuous_scale="RdPu",
        )
        st.plotly_chart(fig, use_container_width=True)

        
        st.subheader("SHAP Summary Plot (Category-Specific)")
        fig2 = plt.figure(figsize=(8, 6))
        try:
            shap.summary_plot(
                shap_array[:, top_idx],
                X_cat_df[top_features],
                feature_names=top_features,
                show=False,
                max_display=10
            )
            st.pyplot(fig2)
        finally:
            plt.close(fig2)
