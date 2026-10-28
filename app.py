
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AgriPrice AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

# =====================================================
# 2. CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #f3f8f1;
    color: #20352a;
}

.main-title {
    color: #176b3a;
    font-size: 42px;
    font-weight: 900;
    margin-bottom: 0;
}

.subtitle {
    color: #587260;
    font-size: 17px;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #103721 0%, #176b3a 60%, #124c2c 100%
    );
    border-right: 2px solid #0b2b19;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 10px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.24);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background-color: #d5f58c;
    border: 2px solid #efffcf;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span {
    color: #103721 !important;
    font-weight: 900 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.35);
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d8e7d4;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(20,80,40,0.06);
}

div[data-testid="stMetric"] label {
    color: #55715d;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: #176b3a;
    font-weight: 850;
}

div.stButton > button {
    background-color: #218448;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 18px;
    font-weight: 800;
    min-height: 45px;
}

div.stButton > button:hover {
    background-color: #125d30;
    color: white;
}

.section-heading {
    color: #176b3a;
    font-size: 25px;
    font-weight: 850;
}

hr {
    border-color: #d5e4d0;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATA AND MODEL
# =====================================================
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)


csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    st.error("No CSV file found in the dataset folder.")
    st.stop()

try:
    df = load_data(str(csv_files[0]))
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

required_columns = [
    "State", "District", "Market", "Commodity",
    "Variety", "Grade", "Min_x0020_Price",
    "Max_x0020_Price", "Modal_x0020_Price"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing required dataset columns: {missing}")
    st.stop()

# Convert prices to numeric values
for column in [
    "Min_x0020_Price",
    "Max_x0020_Price",
    "Modal_x0020_Price"
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Exclude invalid modal prices from price analysis
analysis_df = df.dropna(
    subset=["Commodity", "Market", "Modal_x0020_Price"]
).copy()

if analysis_df.empty:
    st.error("No valid modal-price records are available.")
    st.stop()


# =====================================================
# 4. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 14px;">
        <div style="font-size:50px;">🌾</div>
        <div style="color:white;font-size:27px;font-weight:900;">
            AgriPrice AI
        </div>
        <div style="
            color:#d5f58c;font-size:12px;font-weight:900;
            letter-spacing:1.5px;margin-top:7px;">
            SMART AGRICULTURE
        </div>
        <div style="color:#e0f0e3;font-size:12px;margin-top:8px;">
            Market Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="
        color:#d5f58c;font-size:13px;font-weight:900;
        letter-spacing:1px;margin:10px 0;">
        YOUR WORKSPACE
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🌱 Price Prediction",
            "📊 Market Analysis",
            "🗂️ Dataset Explorer",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.09);
        border:1px solid rgba(255,255,255,0.2);
        border-radius:12px;padding:13px;">
        <div style="color:#d5f58c;font-weight:900;font-size:13px;">
            PROJECT STATUS
        </div>
        <div style="color:white;font-size:13px;margin-top:8px;">
            ● Dataset loaded
        </div>
        <div style="color:white;font-size:13px;margin-top:5px;">
            ● Market analysis ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("AgriPrice AI | Machine Learning Project")


# =====================================================
# 5. MAIN HEADER
# =====================================================
st.markdown(
    '<p class="main-title">🌾 AgriPrice AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Smart Agricultural Market Price '
    'Intelligence & Prediction</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# 6. PRICE PREDICTION PAGE
# =====================================================
if page == "🌱 Price Prediction":

    st.markdown(
        '<p class="section-heading">📍 Market Details</p>',
        unsafe_allow_html=True
    )

    st.write("Choose the market details and enter the price range.")

    c1, c2, c3 = st.columns(3)

    with c1:
        states = sorted(df["State"].dropna().unique().tolist())
        state = st.selectbox("Select State", states)

    state_df = df[df["State"] == state]

    with c2:
        districts = sorted(
            state_df["District"].dropna().unique().tolist()
        )
        district = st.selectbox("Select District", districts)

    district_df = state_df[state_df["District"] == district]

    with c3:
        markets = sorted(
            district_df["Market"].dropna().unique().tolist()
        )
        market = st.selectbox("Select Market", markets)

    c4, c5, c6 = st.columns(3)

    with c4:
        commodities = sorted(
            df["Commodity"].dropna().unique().tolist()
        )
        commodity = st.selectbox("Select Commodity", commodities)

    commodity_df = df[df["Commodity"] == commodity]

    with c5:
        varieties = sorted(
            commodity_df["Variety"].dropna().unique().tolist()
        )
        variety = st.selectbox("Select Variety", varieties)

    variety_df = commodity_df[
        commodity_df["Variety"] == variety
    ]

    with c6:
        grades = sorted(
            variety_df["Grade"].dropna().unique().tolist()
        )
        if not grades:
            grades = sorted(df["Grade"].dropna().unique().tolist())
        grade = st.selectbox("Select Grade", grades)

    st.markdown("---")
    st.markdown(
        '<p class="section-heading">💰 Price Information</p>',
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:
        min_price = st.number_input(
            "Minimum Price (₹)",
            min_value=0.0,
            value=1000.0,
            step=50.0
        )

    with p2:
        max_price = st.number_input(
            "Maximum Price (₹)",
            min_value=0.0,
            value=1500.0,
            step=50.0
        )

    if st.button("🔍 Predict Modal Price", use_container_width=True):

        if min_price > max_price:
            st.error("Minimum price cannot exceed maximum price.")

        elif not MODEL_PATH.exists():
            st.error("Model file not found: models/random_forest.pkl")

        else:
            try:
                model = load_model(str(MODEL_PATH))

                input_data = pd.DataFrame([{
                    "State": state,
                    "District": district,
                    "Market": market,
                    "Commodity": commodity,
                    "Variety": variety,
                    "Grade": grade,
                    "Min_x0020_Price": min_price,
                    "Max_x0020_Price": max_price
                }])

                prediction = float(model.predict(input_data)[0])

                st.markdown("---")
                st.markdown(
                    '<p class="section-heading">📈 Prediction Results</p>',
                    unsafe_allow_html=True
                )

                r1, r2, r3 = st.columns(3)

                r1.metric("Estimated Modal Price", f"₹{prediction:,.2f}")
                r2.metric("Minimum Price", f"₹{min_price:,.2f}")
                r3.metric("Maximum Price", f"₹{max_price:,.2f}")

                st.success(
                    f"Estimated modal price for {commodity}: "
                    f"₹{prediction:,.2f}"
                )

                st.info(
                    "This is a model estimate, not a guaranteed market "
                    "price or a validated future-price forecast."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# =====================================================
# 7. INTERACTIVE MARKET ANALYSIS
# =====================================================
elif page == "📊 Market Analysis":

    st.markdown(
        '<p class="section-heading">📊 Agricultural Market Analysis</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore available commodity prices and compare markets "
        "using interactive filters."
    )

    # Filters
    filter1, filter2, filter3 = st.columns(3)

    state_options = ["All States"] + sorted(
        analysis_df["State"].dropna().unique().tolist()
    )

    commodity_options = ["All Commodities"] + sorted(
        analysis_df["Commodity"].dropna().unique().tolist()
    )

    with filter1:
        selected_state = st.selectbox(
            "Filter by State",
            state_options,
            key="analysis_state"
        )

    with filter2:
        selected_commodity = st.selectbox(
            "Filter by Commodity",
            commodity_options,
            key="analysis_commodity"
        )

    with filter3:
        top_n = st.selectbox(
            "Number of commodities to display",
            [5, 10, 15, 20],
            index=1
        )

    filtered = analysis_df.copy()

    if selected_state != "All States":
        filtered = filtered[filtered["State"] == selected_state]

    if selected_commodity != "All Commodities":
        filtered = filtered[
            filtered["Commodity"] == selected_commodity
        ]

    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # KPI cards
    average_price = filtered["Modal_x0020_Price"].mean()
    highest_price = filtered["Modal_x0020_Price"].max()
    lowest_price = filtered["Modal_x0020_Price"].min()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Valid Records", f"{len(filtered):,}")
    k2.metric("Commodities", f"{filtered['Commodity'].nunique():,}")
    k3.metric("Markets", f"{filtered['Market'].nunique():,}")
    k4.metric("Average Modal Price", f"₹{average_price:,.2f}")

    st.caption(
        f"Highest modal price in filtered data: ₹{highest_price:,.2f} | "
        f"Lowest modal price: ₹{lowest_price:,.2f}"
    )

    st.markdown("---")

    # Chart 1: Top commodities
    st.markdown(
        '<p class="section-heading">🌾 Average Price by Commodity</p>',
        unsafe_allow_html=True
    )

    commodity_summary = (
        filtered.groupby("Commodity")["Modal_x0020_Price"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Average Modal Price",
            "count": "Record Count"
        })
        .sort_values("Average Modal Price", ascending=False)
        .head(top_n)
    )

    if not commodity_summary.empty:
        st.bar_chart(
            commodity_summary.set_index("Commodity")[
                "Average Modal Price"
            ],
            y_label="Average Modal Price (₹)"
        )

        st.dataframe(
            commodity_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Chart 2: Market comparison
    st.markdown(
        '<p class="section-heading">🏪 Compare Market Prices</p>',
        unsafe_allow_html=True
    )

    available_commodities = sorted(
        filtered["Commodity"].dropna().unique().tolist()
    )

    market_commodity = st.selectbox(
        "Choose commodity for market comparison",
        available_commodities,
        key="market_comparison_commodity"
    )

    market_data = filtered[
        filtered["Commodity"] == market_commodity
    ]

    market_summary = (
        market_data.groupby("Market")["Modal_x0020_Price"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Average Modal Price",
            "count": "Record Count"
        })
        .sort_values("Average Modal Price", ascending=False)
        .head(15)
    )

    if not market_summary.empty:
        st.bar_chart(
            market_summary.set_index("Market")["Average Modal Price"],
            y_label="Average Modal Price (₹)"
        )

        st.caption("Showing up to 15 markets with the highest averages.")

        st.dataframe(
            market_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Chart 3: Price distribution
    st.markdown(
        '<p class="section-heading">📈 Modal Price Distribution</p>',
        unsafe_allow_html=True
    )

    st.write(
        "This histogram groups modal prices into ranges to show "
        "how frequently different price levels occur."
    )

    price_values = filtered["Modal_x0020_Price"].dropna()

    if len(price_values) > 0:
        histogram = pd.cut(
            price_values,
            bins=10,
            duplicates="drop"
        ).value_counts(sort=False)

        histogram_df = histogram.rename_axis(
            "Modal Price Range"
        ).reset_index(name="Record Count")

        histogram_df["Modal Price Range"] = (
            histogram_df["Modal Price Range"].astype(str)
        )

        st.bar_chart(
            histogram_df.set_index("Modal Price Range")["Record Count"],
            y_label="Number of Records"
        )

    st.markdown("---")

    # Chart 4: Minimum vs maximum price
    st.markdown(
        '<p class="section-heading">↔️ Minimum and Maximum Prices</p>',
        unsafe_allow_html=True
    )

    range_df = filtered.dropna(
        subset=["Min_x0020_Price", "Max_x0020_Price"]
    ).copy()

    if not range_df.empty:
        range_summary = (
            range_df.groupby("Commodity")[
                ["Min_x0020_Price", "Max_x0020_Price"]
            ]
            .mean()
            .rename(columns={
                "Min_x0020_Price": "Average Minimum Price",
                "Max_x0020_Price": "Average Maximum Price"
            })
            .sort_values("Average Maximum Price", ascending=False)
            .head(top_n)
        )

        st.bar_chart(
            range_summary,
            y_label="Average Price (₹)"
        )

    # Download data
    st.markdown("---")
    st.subheader("⬇️ Download Analysis Data")

    st.download_button(
        "Download Filtered Market Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="agriprice_market_analysis.csv",
        mime="text/csv"
    )

    st.caption(
        "Analysis describes the available dataset only. Averages can be "
        "affected by sample size, outliers and differences between markets. "
        "They are not guaranteed current prices."
    )


# =====================================================
# 8. DATASET EXPLORER
# =====================================================
elif page == "🗂️ Dataset Explorer":

    st.markdown(
        '<p class="section-heading">🗂️ Dataset Explorer</p>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Commodities", f"{df['Commodity'].nunique():,}")
    m3.metric("Markets", f"{df['Market'].nunique():,}")

    selected = st.selectbox(
        "Filter by Commodity",
        ["All Commodities"] +
        sorted(df["Commodity"].dropna().unique().tolist()),
        key="explorer_filter"
    )

    if selected == "All Commodities":
        explorer_df = df.copy()
    else:
        explorer_df = df[df["Commodity"] == selected].copy()

    st.dataframe(
        explorer_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    summary = (
        explorer_df.groupby("Commodity")["Modal_x0020_Price"]
        .agg(["count", "mean", "min", "max"])
        .round(2)
        .reset_index()
    )

    summary.columns = [
        "Commodity", "Record Count", "Average Modal Price",
        "Minimum Modal Price", "Maximum Modal Price"
    ]

    st.subheader("Commodity Price Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=explorer_df.to_csv(index=False).encode("utf-8"),
        file_name="agriprice_filtered_data.csv",
        mime="text/csv"
    )


# =====================================================
# 9. ABOUT PROJECT
# =====================================================
elif page == "ℹ️ About Project":

    st.markdown(
        '<p class="section-heading">🌱 About AgriPrice AI</p>',
        unsafe_allow_html=True
    )

    st.write("""
    AgriPrice AI is an agricultural market price intelligence application
    developed using Python, Pandas, Scikit-learn and Streamlit.

    It uses a trained Random Forest model to estimate modal prices from
    commodity and market details and entered minimum/maximum prices.
    """)

    st.subheader("Technology Stack")

    st.markdown("""
    - **Python:** Core programming language
    - **Pandas:** Data processing and analysis
    - **Scikit-learn:** Machine learning and preprocessing
    - **Random Forest:** Trained price estimation model
    - **Streamlit:** Interactive web interface
    - **Joblib:** Model saving and loading
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze agricultural market price data.
    - Compare commodity prices across markets.
    - Estimate modal prices with machine learning.
    - Present market information using interactive charts.
    """)

    st.warning(
        "The model has not been validated as a future-price forecasting "
        "system. Its current evaluation used a random train/test split."
    )


# =====================================================
# 10. FOOTER
# =====================================================
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#587260;
        font-size:13px;
        font-weight:700;
        padding:10px 0;">
        🌾 AgriPrice AI | Smart Agriculture | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
