# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Laptop Pricing Dashboard", layout="wide", initial_sidebar_state="expanded")

DATA_PATH = r"C:\Users\Rohan das\Downloads\laptop_prices.csv"  # <<-- uses your absolute path

# -----------------------
# STYLES (dark teal/blue) - Enhanced with better spacing
# -----------------------
PRIMARY = "#1abc9c"   # teal
SECONDARY = "#3498db" # blue
BG = "#0f1724"        # near-black blue
CARD = "#111421"
CARD2 = "#111827"
TEXT = "#E6EEF3"

st.markdown(
    f"""
    <style>
    /* Page background */
    .reportview-container, .main {{
        background-color: {BG};
        color: {TEXT};
    }}
    /* Sidebar */
    .css-1d391kg .css-1d391kg {{
        background-color: {BG};
    }}
    .stSidebar {{
        background-color: {BG};
    }}
    /* Cards */
    .metric-card {{
        background: linear-gradient(180deg, {CARD} 0%, {CARD2} 100%);
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 12px;
    }}
    .metric-title {{color: #9fb9b1; font-size:13px; margin-bottom:6px;}}
    .metric-value {{color: {PRIMARY}; font-size:28px; font-weight:700;}}
    /* small text */
    .small {{color: #9fb9b1; font-size:12px;}}
    /* make sidebar expander content scrollable (dropdown-like) */
    [data-testid="stSidebar"] .stExpander .stExpanderContent {{
        max-height: 220px;
        overflow-y: auto;
        padding-right: 8px;
    }}
    /* heading style */
    .dashboard-title {{ font-size: 26px; color: {PRIMARY}; text-align:center; margin-bottom:6px; }}
    .dashboard-sub {{ color: #9fb9b1; text-align:center; margin-top: -6px; margin-bottom:14px; }}
    /* bottom-right name */
    .owner-name {{
        position: fixed;
        right: 18px;
        bottom: 8px;
        color: #9fb9b1;
        font-size: 12px;
    }}
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {CARD};
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        margin-right: 0;
    }}
    .stTabs [aria-selected="true"] {{
        background: {PRIMARY};
        color: {BG} !important;
    }}
    /* Better spacing */
    .stPlotlyChart, .stDataFrame {{
        margin-top: 8px;
    }}
    /* Custom expander headers */
    .streamlit-expanderHeader {{
        font-weight: 600;
        color: {PRIMARY};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Heading (with better visual hierarchy)
# -----------------------
st.markdown(f"<div class='dashboard-title'>💻 Laptop Pricing Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown(f"<div class='dashboard-sub'>Advanced analytics & market insights</div>", unsafe_allow_html=True)

# -----------------------
# DATA LOAD & PREPROCESSING
# -----------------------
@st.cache_data
def load_data(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(p, encoding="ISO-8859-1")
    
    # Enhanced preprocessing
    if 'Ram' in df.columns:
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').astype(float)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)
    if 'Inches' in df.columns:
        df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
    
    # Calculate PPI (Pixels Per Inch) if screen dimensions available
    if all(col in df.columns for col in ['ScreenW', 'ScreenH', 'Inches']):
        df['PPI'] = ((df['ScreenW']**2 + df['ScreenH']**2)**0.5 / df['Inches']).round(1)
    
    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# -----------------------
# SIDEBAR FILTERS (improved organization)
# -----------------------
st.sidebar.markdown("<h3 style='color: #BFECE1'>🔍 Filter Laptops</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Helper functions (unchanged from your original)
def _init_checkbox_keys(label, values, all_default=True):
    all_key = f"all__{label}"
    if all_key not in st.session_state:
        st.session_state[all_key] = all_default
    for v in values:
        key = f"{label}__{v}"
        if key not in st.session_state:
            st.session_state[key] = all_default

def _on_all_change(label, values):
    all_key = f"all__{label}"
    all_state = st.session_state.get(all_key, False)
    for v in values:
        key = f"{label}__{v}"
        st.session_state[key] = all_state

def checkbox_group(label, series):
    values = sorted(series.dropna().unique().tolist())
    if not values:
        return []

    _init_checkbox_keys(label, values, all_default=True)
    all_key = f"all__{label}"
    
    with st.sidebar.expander(f"▸ {label}", expanded=False):
        st.checkbox(f"All {label}", value=st.session_state[all_key], key=all_key,
                    on_change=_on_all_change, args=(label, values))

        selected = []
        for v in values:
            key = f"{label}__{v}"
            checked = st.checkbox(str(v), value=st.session_state[key], key=key)
            if checked:
                selected.append(v)
    return selected

# Main filters with better organization
with st.sidebar.expander("▸ Brand & Type", expanded=True):
    company_sel = checkbox_group("Company", df["Company"] if "Company" in df.columns else pd.Series([]))
    type_sel = checkbox_group("TypeName", df["TypeName"] if "TypeName" in df.columns else pd.Series([]))

with st.sidebar.expander("▸ Specifications", expanded=True):
    os_sel = checkbox_group("OS", df["OS"] if "OS" in df.columns else pd.Series([]))
    
    # Enhanced RAM filter with dynamic options
    if 'Ram' in df.columns:
        ram_options = sorted(df['Ram'].dropna().unique())
        ram_sel = st.multiselect("RAM (GB)", options=ram_options, default=ram_options)
    
    # Enhanced storage filter
    if 'PrimaryStorage' in df.columns:
        storage_options = sorted(df['PrimaryStorage'].dropna().unique())
        storage_sel = st.slider("Minimum Storage (GB)", 
                               min_value=int(min(storage_options)), 
                               max_value=int(max(storage_options)), 
                               value=int(min(storage_options)))

# Price slider with better formatting
min_price = int(np.nanmin(df["Price_euros"])) if "Price_euros" in df.columns and not df["Price_euros"].isna().all() else 0
max_price = int(np.nanmax(df["Price_euros"])) if "Price_euros" in df.columns and not df["Price_euros"].isna().all() else 1
price_range = st.sidebar.slider("💰 Price Range (€)", min_price, max_price, (min_price, max_price), 
                               help="Adjust the price range to filter laptops")

# Display features toggle
st.sidebar.markdown("---")
st.sidebar.markdown("**Display Features**")
touch_toggle = st.sidebar.radio("Touchscreen", options=["All","Yes","No"], horizontal=True)
ips_toggle = st.sidebar.radio("IPS Panel", options=["All","Yes","No"], horizontal=True)
retina_toggle = st.sidebar.radio("Retina Display", options=["All","Yes","No"], horizontal=True) if "RetinaDisplay" in df.columns else None

# Reset button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    keys_to_clear = [k for k in st.session_state.keys() if any(prefix in k for prefix in ["all__Company", "Company__", "all__TypeName", "TypeName__", "all__OS", "OS__"])]
    for kk in keys_to_clear:
        del st.session_state[kk]
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# -----------------------
# APPLY FILTERS (enhanced with new filters)
# -----------------------
df_filtered = df.copy()

def apply_list_filter(df_local, column, selected_list):
    if column not in df_local.columns:
        return df_local
    if selected_list:
        available = sorted(df_local[column].dropna().unique().tolist())
        if set(selected_list) == set(available):
            return df_local
        else:
            return df_local[df_local[column].isin(selected_list)]
    return df_local

df_filtered = apply_list_filter(df_filtered, "Company", company_sel)
df_filtered = apply_list_filter(df_filtered, "TypeName", type_sel)
df_filtered = apply_list_filter(df_filtered, "OS", os_sel)

if "Price_euros" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Price_euros"].between(price_range[0], price_range[1], inclusive="both")]

if 'Ram' in df_filtered.columns and 'ram_sel' in locals():
    df_filtered = df_filtered[df_filtered['Ram'].isin(ram_sel)]

if 'PrimaryStorage' in df_filtered.columns and 'storage_sel' in locals():
    df_filtered = df_filtered[df_filtered['PrimaryStorage'] >= storage_sel]

# Display feature filters
display_filters = {
    "Touchscreen": touch_toggle,
    "IPSpanel": ips_toggle,
    "RetinaDisplay": retina_toggle if retina_toggle else None
}

for feature, value in display_filters.items():
    if value != "All" and value is not None and feature in df_filtered.columns:
        if value == "Yes":
            df_filtered = df_filtered[df_filtered[feature].astype(str).str.lower().isin(["yes","true","1"])]
        else:
            df_filtered = df_filtered[~df_filtered[feature].astype(str).str.lower().isin(["yes","true","1"])]

# Safety check
if df_filtered.empty:
    st.markdown("<div class='metric-card'><div class='metric-title'>No matching laptops found</div><div class='metric-value'>0</div><div class='small'>Try expanding your filters or select 'All' options.</div></div>", unsafe_allow_html=True)
    st.stop()

# -----------------------
# DASHBOARD CONTENT - Using Tabs for better organization
# -----------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Deep Dive", "🤖 Price Predictor"])

with tab1:
    # Enhanced metrics with delta indicators
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df_filtered)
    total_all = len(df)
    delta_total = f"{((total/total_all)*100-100):+.1f}%" if total_all > 0 else "N/A"
    
    avg_price = df_filtered["Price_euros"].mean() if "Price_euros" in df_filtered.columns else np.nan
    avg_all = df["Price_euros"].mean() if "Price_euros" in df.columns else np.nan
    delta_avg = f"{(avg_price/avg_all*100-100):+.1f}%" if not np.isnan(avg_all) else "N/A"
    
    median_price = df_filtered["Price_euros"].median() if "Price_euros" in df_filtered.columns else np.nan
    max_price = df_filtered["Price_euros"].max() if "Price_euros" in df_filtered.columns else np.nan
    
    # Most common brand with percentage
    most_common_brand = "N/A"
    brand_pct = ""
    if "Company" in df_filtered.columns and not df_filtered["Company"].dropna().empty:
        mode_series = df_filtered["Company"].mode()
        if not mode_series.empty:
            most_common_brand = mode_series.iloc[0]
            brand_count = df_filtered[df_filtered["Company"] == most_common_brand].shape[0]
            brand_pct = f"({(brand_count/total)*100:.1f}%)"
    
    col1.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Filtered Laptops</div>
            <div class='metric-value'>{total}</div>
            <div class='small'>vs. {total_all} total • {delta_total}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Avg Price (€)</div>
            <div class='metric-value'>{avg_price:,.0f}</div>
            <div class='small'>vs. {avg_all:,.0f} avg • {delta_avg}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Median Price (€)</div>
            <div class='metric-value'>{median_price:,.0f}</div>
            <div class='small'>Max: {max_price:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col4.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Top Brand</div>
            <div class='metric-value'>{most_common_brand}</div>
            <div class='small'>{brand_pct}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 1: Price distribution + Brand comparison
    r1c1, r1c2 = st.columns([2, 3])
    
    with r1c1:
        st.subheader("Price Distribution")
        if "Price_euros" in df_filtered.columns:
            fig_hist = px.histogram(
                df_filtered, x="Price_euros", nbins=30, 
                title="", marginal="box", 
                labels={"Price_euros":"Price (€)"},
                color_discrete_sequence=[PRIMARY],
                hover_data=["Company", "TypeName"]
            )
            fig_hist.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, 
                font_color=TEXT,
                hoverlabel=dict(bgcolor=CARD, font_color=TEXT)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with r1c2:
        st.subheader("Brand Comparison")
        if "Company" in df_filtered.columns and "Price_euros" in df_filtered.columns:
            brand_stats = df_filtered.groupby("Company").agg(
                Avg_Price=("Price_euros", "mean"),
                Count=("Price_euros", "size"),
                Median_Price=("Price_euros", "median")
            ).sort_values("Avg_Price", ascending=False).reset_index()
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add price bars
            fig.add_trace(go.Bar(
                x=brand_stats["Company"],
                y=brand_stats["Avg_Price"],
                name="Avg Price (€)",
                marker_color=PRIMARY,
                text=brand_stats["Avg_Price"].round(0),
                textposition='outside'
            ))
            
            # Add count line
            fig.add_trace(go.Scatter(
                x=brand_stats["Company"],
                y=brand_stats["Count"],
                name="Count",
                mode='lines+markers',
                yaxis="y2",
                line=dict(color=SECONDARY, width=2),
                marker=dict(size=8, color=SECONDARY)
            ))
            
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, 
                font_color=TEXT,
                yaxis=dict(title="Avg Price (€)", color=PRIMARY),
                yaxis2=dict(
                    title="Count",
                    overlaying="y",
                    side="right",
                    color=SECONDARY
                ),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Scatter matrix for key features
    st.subheader("Feature Relationships")
    scatter_features = []
    if "Ram" in df_filtered.columns: scatter_features.append("Ram")
    if "Inches" in df_filtered.columns: scatter_features.append("Inches")
    if "Weight" in df_filtered.columns: scatter_features.append("Weight")
    if "PPI" in df_filtered.columns: scatter_features.append("PPI")
    if "Price_euros" in df_filtered.columns: scatter_features.append("Price_euros")
    
    if len(scatter_features) >= 3:  # Need at least 3 features for interesting matrix
        fig_matrix = px.scatter_matrix(
            df_filtered,
            dimensions=scatter_features,
            color="Company" if "Company" in df_filtered.columns else None,
            hover_data=["Product"],
            title="",
            height=700
        )
        fig_matrix.update_layout(
            plot_bgcolor=BG, paper_bgcolor=BG, 
            font_color=TEXT,
            hoverlabel=dict(bgcolor=CARD, font_color=TEXT)
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
    else:
        st.info("Insufficient numeric columns for scatter matrix")

with tab2:
    # Deep dive analysis
    st.subheader("Detailed Specifications Analysis")
    
    # Row 1: RAM vs Price with storage type
    r1c1, r1c2 = st.columns(2)
    
    with r1c1:
        st.markdown("**RAM vs Price by Storage Type**")
        if all(col in df_filtered.columns for col in ["Ram", "Price_euros", "PrimaryStorageType"]):
            fig = px.scatter(
                df_filtered,
                x="Ram", y="Price_euros",
                color="PrimaryStorageType",
                size="PrimaryStorage" if "PrimaryStorage" in df_filtered.columns else None,
                hover_data=["Company", "Product", "CPU_model"],
                log_y=True,
                labels={"Price_euros": "Price (€)", "Ram": "RAM (GB)"},
                height=500
            )
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, 
                font_color=TEXT,
                hoverlabel=dict(bgcolor=CARD, font_color=TEXT)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with r1c2:
        st.markdown("**Screen Technology Impact**")
        if all(col in df_filtered.columns for col in ["Price_euros", "IPSpanel", "Touchscreen", "RetinaDisplay"]):
            tech_cols = ["IPSpanel", "Touchscreen", "RetinaDisplay"]
            tech_df = df_filtered.copy()
            
            # Convert tech features to numeric for aggregation
            for col in tech_cols:
                if col in tech_df.columns:
                    tech_df[col] = tech_df[col].astype(str).str.lower().isin(["yes","true","1"]).astype(int)
            
            # Calculate average price by tech combinations
            tech_impact = tech_df.groupby(tech_cols)["Price_euros"].mean().reset_index()
            tech_impact['Tech_Combo'] = tech_impact.apply(lambda row: 
                " ".join([col for col in tech_cols if row[col] == 1]), axis=1)
            
            fig = px.bar(
                tech_impact.sort_values("Price_euros", ascending=False),
                x="Tech_Combo", y="Price_euros",
                color="Price_euros",
                color_continuous_scale=px.colors.sequential.Teal,
                labels={"Price_euros": "Avg Price (€)"},
                height=500
            )
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, 
                font_color=TEXT,
                xaxis_title="Technology Combination",
                hoverlabel=dict(bgcolor=CARD, font_color=TEXT)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    # Row 2: CPU and GPU analysis
    st.markdown("---")
    r2c1, r2c2 = st.columns(2)
    
    with r2c1:
        st.markdown("**Top CPU Models by Price**")
        if all(col in df_filtered.columns for col in ["CPU_model", "Price_euros"]):
            cpu_stats = df_filtered.groupby("CPU_model").agg(
                Avg_Price=("Price_euros", "mean"),
                Count=("Price_euros", "size")
            ).sort_values("Avg_Price", ascending=False).head(15).reset_index()
            
            fig = px.bar(
                cpu_stats,
                x="CPU_model", y="Avg_Price",
                color="Count",
                text="Avg_Price",
                labels={"Avg_Price": "Avg Price (€)", "CPU_model": "CPU Model"},
                height=500
            )
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, 
                font_color=TEXT,
                hoverlabel=dict(bgcolor=CARD, font_color=TEXT),
                xaxis_tickangle=-45
            )
            fig.update_traces(texttemplate='%{text:.0f}€', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with r2c2:
        st.markdown("**GPU Performance Tier Analysis**")
        if all(col in df_filtered.columns for col in ["GPU_model", "Price_euros"]):
            # Simple GPU tier classification (this could be enhanced with a proper GPU performance database)
            df_filtered['GPU_Tier'] = df_filtered['GPU_model'].str.extract(r'(\d+)', expand=False)
            df_filtered['GPU_Tier'] = pd.to_numeric(df_filtered['GPU_Tier'])
            df_filtered['GPU_Tier'] = pd.cut(
                df_filtered['GPU_Tier'],
                bins=[0, 30, 50, 70, 90, 110, 130, 150, 200, 300, 1000],
                labels=['<30', '30-50', '50-70', '70-90', '90-110', '110-130', '130-150', '150-200', '200-300', '300+']
            )
            
            gpu_tier_stats = df_filtered.groupby("GPU_Tier").agg(
                Avg_Price=("Price_euros", "mean"),
                Count=("Price_euros", "size")
            ).reset_index()
            
            fig = px.scatter(
                gpu_tier_stats,
                x="GPU_Tier", y="Avg_Price",
                size="Count",
                color="Count",
                hover_name="GPU_Tier",
                labels={"Avg_Price": "Avg Price (€)", "GPU_Tier": "GPU Performance Tier"},
                height=500
            )
            fig.update_layout(
                plot_bgcolor=BG, paper_bgcolor=BG, 
                font_color=TEXT,
                hoverlabel=dict(bgcolor=CARD, font_color=TEXT)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("---")
    st.markdown("**Feature Correlation Matrix**")
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1 and "Price_euros" in numeric_cols:
        corr_matrix = df_filtered[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale=px.colors.sequential.Teal,
            aspect="auto"
        )
        fig.update_layout(
            plot_bgcolor=BG, paper_bgcolor=BG, 
            font_color=TEXT,
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Machine Learning Price Prediction
    st.subheader("Laptop Price Predictor")
    
    # Only run prediction if we have enough data
    if len(df_filtered) < 50:
        st.warning("Need at least 50 records in filtered dataset to build prediction model")
    else:
        # Prepare data for modeling
        @st.cache_resource
        def train_model(data):
            # Select features - adjust based on your dataset
            features = [
                'Company', 'TypeName', 'Ram', 'Inches', 'Weight',
                'PrimaryStorage', 'PrimaryStorageType', 'CPU_model', 'GPU_model'
            ]
            
            # Filter to available features
            features = [f for f in features if f in data.columns]
            
            # Create feature matrix
            X = data[features].copy()
            y = data['Price_euros']
            
            # Encode categorical features
            cat_cols = X.select_dtypes(include=['object']).columns
            for col in cat_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            return model, features, mae
        
        model, features, mae = train_model(df_filtered)
        
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Model Performance (MAE)</div>
                <div class='metric-value'>{mae:,.0f} €</div>
                <div class='small'>Mean Absolute Error on test set</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Predict Price for Custom Configuration**")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            inputs = {}
            if 'Company' in features:
                inputs['Company'] = col1.selectbox("Brand", options=sorted(df_filtered['Company'].unique()))
            if 'TypeName' in features:
                inputs['TypeName'] = col2.selectbox("Type", options=sorted(df_filtered['TypeName'].unique()))
            if 'Ram' in features:
                inputs['Ram'] = col1.slider("RAM (GB)", 
                                          min_value=int(df_filtered['Ram'].min()), 
                                          max_value=int(df_filtered['Ram'].max()), 
                                          value=int(df_filtered['Ram'].median()))
            if 'Inches' in features:
                inputs['Inches'] = col2.slider("Screen Size (Inches)", 
                                             min_value=float(df_filtered['Inches'].min()), 
                                             max_value=float(df_filtered['Inches'].max()), 
                                             value=float(df_filtered['Inches'].median()))
            if 'Weight' in features:
                inputs['Weight'] = col1.slider("Weight (kg)", 
                                             min_value=float(df_filtered['Weight'].min()), 
                                             max_value=float(df_filtered['Weight'].max()), 
                                             value=float(df_filtered['Weight'].median()))
            if 'PrimaryStorage' in features:
                inputs['PrimaryStorage'] = col2.slider("Primary Storage (GB)", 
                                                     min_value=int(df_filtered['PrimaryStorage'].min()), 
                                                     max_value=int(df_filtered['PrimaryStorage'].max()), 
                                                     value=int(df_filtered['PrimaryStorage'].median()))
            if 'PrimaryStorageType' in features:
                inputs['PrimaryStorageType'] = col1.selectbox("Storage Type", 
                                                            options=sorted(df_filtered['PrimaryStorageType'].dropna().unique()))
            if 'CPU_model' in features:
                inputs['CPU_model'] = col2.selectbox("CPU Model", 
                                                   options=sorted(df_filtered['CPU_model'].dropna().unique()))
            if 'GPU_model' in features:
                inputs['GPU_model'] = col1.selectbox("GPU Model", 
                                                   options=sorted(df_filtered['GPU_model'].dropna().unique()))
            
            submitted = st.form_submit_button("Predict Price")
            
            if submitted:
                # Prepare input data
                input_df = pd.DataFrame([inputs])
                
                # Encode categorical variables same as training
                cat_cols = input_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    le = LabelEncoder()
                    le.classes_ = np.array(sorted(df_filtered[col].astype(str).unique()))
                    input_df[col] = le.transform(input_df[col].astype(str))
                
                # Make prediction
                prediction = model.predict(input_df[features])[0]
                
                # Show result
                st.markdown(f"""
                    <div class='metric-card' style='margin-top:20px;'>
                        <div class='metric-title'>Predicted Price</div>
                        <div class='metric-value'>{prediction:,.0f} €</div>
                        <div class='small'>± {mae:,.0f} € (estimated error)</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show similar laptops
                st.markdown("**Similar Laptops in Dataset**")
                similar = df_filtered.copy()
                
                # Calculate similarity score (simple version)
                for col in inputs:
                    if col in similar.columns:
                        if similar[col].dtype == 'object':
                            similar[col+'_match'] = (similar[col] == inputs[col]).astype(int)
                        else:
                            similar[col+'_match'] = 1 - (abs(similar[col] - inputs[col]) / (similar[col].max() - similar[col].min()))
                
                match_cols = [c for c in similar.columns if c.endswith('_match')]
                similar['match_score'] = similar[match_cols].mean(axis=1)
                
                st.dataframe(
                    similar.sort_values('match_score', ascending=False)
                    .head(5)[['Company', 'Product', 'Ram', 'PrimaryStorage', 'Price_euros']]
                    .rename(columns={'Price_euros': 'Price (€)'}),
                    hide_index=True
                )

# -----------------------
# DATA TABLE (in all tabs)
# -----------------------
with st.expander("📋 Show Filtered Data (first 200 rows)"):
    st.dataframe(df_filtered.head(200), use_container_width=True)

# -----------------------
# FOOTER
# -----------------------
st.markdown("---", unsafe_allow_html=True)
st.markdown("<div class='owner-name'>— Rohan's Laptop Analytics</div>", unsafe_allow_html=True)