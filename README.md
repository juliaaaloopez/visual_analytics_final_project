# Sephora Popularity Analytics

## Project Overview
This repository combines exploratory analysis, modeling, and an interactive dashboard to understand what drives product popularity in Sephora's catalog. The workflow:
- Cleans and explores the catalog in depth, surfacing price, engagement, ingredient, and highlight trends.
- Trains an explainable machine learning model that predicts whether a SKU will be popular, then interprets decisions with SHAP.
- Packages the insights into a multi-page Streamlit app so merchandisers can experiment with filters, examine drivers, and simulate new product concepts.

## Repository Structure
| Path | Description |
| --- | --- |
| `Final_project.ipynb` | End-to-end notebook covering EDA, feature engineering, model training/tuning, evaluation, and SHAP explainability. |
| `app_utils.py` | Shared helpers for loading data, preparing feature lists, computing uplift metrics, and caching model artifacts for Streamlit. |
| `introduction_and_overview.py` | Streamlit landing page that frames the business problem and highlights top-line metrics. |
| `pages/1_EDA_Dashboard.py` | Streamlit page for interactive exploratory analysis (filters, distributions, correlation views). |
| `pages/2_Popularity_Drivers.py` | Streamlit page focused on SHAP-based feature importance and per-category explanations. |
| `pages/3_Ingredient_and_Highlights.py` | Streamlit page that dives into ingredient/highlight prevalence, combinations, and category-specific insights. |
| `pages/4_Product_Prediction.py` | Streamlit page that simulates new SKUs, predicts popularity, and surfaces SHAP explanations for each scenario. |
| `data/` | Raw and processed CSV files (e.g., `products_final.csv`, ingredient mappings) required for analysis. |
| `models/` | Serialized model bundles (pipeline, explainer, SHAP values, transformed matrices) used by the app. |

## How to Run the Notebook
1. **Install dependencies** (in a virtual environment is recommended):
   ```bash
   pip install pandas numpy scikit-learn shap matplotlib seaborn plotly jupyter
   ```
2. **Open the notebook** `Final_project.ipynb` in JupyterLab, VS Code, or any Jupyter-compatible IDE.
3. **Run cells sequentially**. Expect to see:
   - Exploratory visuals (distribution plots, correlation heatmaps, uplift tables).
   - Feature engineering and modeling outputs (train/test metrics, feature importances).
   - SHAP explainability charts (global importance, dependence plots, per-category slices).

## How to Run the Streamlit App
1. **Create/activate an environment** (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Streamlit** from the project root:
   ```bash
   streamlit run introduction_and_overview.py
   ```
4. **Navigate via the sidebar** to access all pages (Overview, EDA Dashboard, Popularity Drivers, Ingredient & Highlight Insights, Product Prediction).

## Requirements
- **Python**: 3.9+ recommended.
- **Core libraries**: `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `seaborn`, `streamlit`, `plotly`.
- A modern browser for viewing the Streamlit app.

Feel free to adapt the workflow with new datasets or extend the Streamlit pages for additional business questions.
