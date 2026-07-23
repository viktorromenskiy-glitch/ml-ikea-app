"""
IKEA Furniture Price Predictor — Interactive Demo (Streamlit)
Random Forest + Optuna (Bayesian hyperparameter search), R²=0.83 on held-out test data.
Full methodology, statistical validation, and honest limitations:
https://github.com/viktorromenskiy-glitch/ikea-pricing-analysis
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IKEA Price Predictor", page_icon="🪑", layout="centered")

# ------------------------------------------------------------------
# Load model + lookup tables (precomputed from training data)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("ikea_price_model.joblib")

@st.cache_resource
def load_lookup():
    with open("lookup_tables.json", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
LOOKUP = load_lookup()

CATEGORY_PRICE_LEVEL = LOOKUP["category_price_level"]
DESIGNER_FREQ = LOOKUP["designer_freq"]
DESIGNER_FREQ_MEDIAN = LOOKUP["designer_freq_median"]
CATEGORIES = sorted(CATEGORY_PRICE_LEVEL.keys())

ASSEMBLY_MAP = {
    'Bar furniture': 0.2, 'Beds': 0.7, 'Bookcases & shelving units': 0.5,
    'Cabinets & cupboards': 0.5, 'Café furniture': 0.2, "Children's furniture": 0.4,
    'Chairs': 0.3, 'Chests of drawers & drawer units': 0.5, 'Nursery furniture': 0.4,
    'Outdoor furniture': 0.3, 'Room dividers': 0.2, 'Sideboards, buffets & console tables': 0.5,
    'Sofas & armchairs': 0.6, 'Tables & desks': 0.4, 'TV & media furniture': 0.5,
    'Trolleys': 0.2, 'Wardrobes': 0.8,
}
PREMIUM_CATEGORIES = {'Wardrobes', 'Sofas & armchairs', 'Beds', 'Cabinets & cupboards', 'TV & media furniture'}
PREMIUM_MATERIAL_MARKERS = ["solid wood", "oak", "walnut", "leather", "steel", "glass"]

NUMERIC_FEATURES = ['depth', 'height', 'width', 'desc_length', 'desc_word_count',
                     'premium_materials_count', 'category_price_level', 'designer_freq', 'assembly_complexity']
BINARY_FEATURES = ['is_team', 'has_other_colors', 'has_old_price', 'is_premium_category']
CATEGORICAL_FEATURES = ['category']


def predict_price(category, depth, height, width, description, designer, has_other_colors, has_discount):
    description = description or ""
    designer = (designer or "").strip()

    desc_length = len(description)
    desc_word_count = len(description.split())
    desc_lower = description.lower()
    premium_materials_count = sum(1 for m in PREMIUM_MATERIAL_MARKERS if m in desc_lower)

    designer_freq = DESIGNER_FREQ.get(designer, DESIGNER_FREQ_MEDIAN)
    is_team = 1 if any(sep in designer for sep in ["/", "&", ","]) else 0

    category_price_level = CATEGORY_PRICE_LEVEL.get(category, float(np.median(list(CATEGORY_PRICE_LEVEL.values()))))
    assembly_complexity = ASSEMBLY_MAP.get(category, 0.4)
    is_premium_category = 1 if category in PREMIUM_CATEGORIES else 0

    row = pd.DataFrame([{
        'depth': depth, 'height': height, 'width': width,
        'desc_length': desc_length, 'desc_word_count': desc_word_count,
        'premium_materials_count': premium_materials_count,
        'category_price_level': category_price_level,
        'designer_freq': designer_freq,
        'assembly_complexity': assembly_complexity,
        'is_team': is_team,
        'has_other_colors': 1 if has_other_colors else 0,
        'has_old_price': 1 if has_discount else 0,
        'is_premium_category': is_premium_category,
        'category': category,
    }])[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]

    log_pred = model.predict(row)[0]
    price = float(np.expm1(log_pred))

    details = {
        "Category price level": f"{category_price_level:,.0f} SR",
        "Assembly complexity": f"{assembly_complexity:.1f}",
        "Premium category": "Yes" if is_premium_category else "No",
        "Description length / words": f"{desc_length} chars / {desc_word_count} words",
        "Premium material mentions": premium_materials_count,
        "Designer frequency": f"{designer_freq:.0f}" + (" (known)" if designer in DESIGNER_FREQ else " (unknown, using median)"),
        "Team-designed": "Yes" if is_team else "No",
    }
    return price, details


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🪑 IKEA Furniture Price Predictor")
st.markdown(
    """
    Random Forest + Optuna (Bayesian hyperparameter search) — **R²=0.83** on held-out test data,
    trained on 2,962 unique IKEA products (Saudi Arabia market,
    [TidyTuesday dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-11-03)).

    This demo derives most model features automatically from a few natural inputs, the same way
    the training pipeline does — type a description or a designer name and watch the engineered
    features update in the breakdown below.
    """
)
with st.expander("⚠️ Honest limitations (read before trusting a prediction)"):
    st.markdown(
        """
        This model systematically **underprices premium items** (>2000 SR) and has
        **heteroscedastic errors** (accuracy degrades at the high end). It's a validated portfolio
        project demonstrating rigorous ML methodology — 9 statistical hypothesis tests, 3 independent
        leakage checks, group and point-level ablation testing — not a production pricing tool.

        Full methodology and an honest account of every modeling decision (including one feature
        flagged by a reviewer, tested at two different statistical power levels, and removed only
        after a stale-cache bug was caught and fixed):
        [github.com/viktorromenskiy-glitch/ikea-pricing-analysis](https://github.com/viktorromenskiy-glitch/ikea-pricing-analysis)
        """
    )

st.divider()

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index("Bookcases & shelving units"))
    depth = st.slider("Depth (cm)", 1, 250, 40)
    height = st.slider("Height (cm)", 1, 700, 100)
    width = st.slider("Width (cm)", 1, 420, 80)
with col2:
    description = st.text_area(
        "Short description",
        placeholder="e.g. 'Bookcase, solid oak veneer, adjustable shelves'",
        height=100,
    )
    designer = st.text_input(
        "Designer name",
        placeholder="e.g. 'IKEA of Sweden' or 'Ehlén Johansson/IKEA of Sweden' for a team",
    )
    has_other_colors = st.checkbox("Available in other colors")
    has_discount = st.checkbox("Currently discounted (has old price)")

st.divider()

examples = {
    "— pick an example (optional) —": None,
    "Wardrobe, solid wood, team-designed": ("Wardrobes", 60, 236, 150, "Sliding door wardrobe, solid wood frame, mirror doors", "Ehlén Johansson/IKEA of Sweden", True, False),
    "Simple bookcase, particleboard": ("Bookcases & shelving units", 30, 180, 80, "Simple bookcase, particleboard", "IKEA of Sweden", False, False),
    "Leather sofa, discounted": ("Sofas & armchairs", 90, 85, 220, "3-seat sofa, leather upholstery, premium fabric options", "Francis Cayouette", True, True),
}
choice = st.selectbox("Try an example", list(examples.keys()))

if st.button("Predict price", type="primary", use_container_width=True):
    if examples[choice] is not None:
        category, depth, height, width, description, designer, has_other_colors, has_discount = examples[choice]
    price, details = predict_price(category, depth, height, width, description, designer, has_other_colors, has_discount)
    st.success(f"### Predicted price: **{price:,.0f} SR**")
    st.markdown("**Auto-derived features used by the model:**")
    st.table(pd.DataFrame(details.items(), columns=["Feature", "Value"]))

st.caption("Viktor Romensky · Data Science Portfolio Project")
