import streamlit as st
import pandas as pd
import pickle
import json
import os
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Configure page settings
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main background and text colors */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00cc66;
        --accent-color: #ff6b6b;
        --background-color: #f8f9fa;
        --text-color: #2c3e50;
    }
    
    /* Hide Streamlit default UI elements */
    #MainMenu, footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #00cc66 100%);
        padding: 40px 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Subheader styling */
    .section-header {
        font-size: 1.5em;
        font-weight: 600;
        color: #0066cc;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 3px solid #00cc66;
        display: inline-block;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #0066cc;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Info message styling */
    .info-message {
        background-color: #e3f2fd;
        border-left: 4px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.95em;
    }
    
    /* Success message styling */
    .success-message {
        background-color: #e8f5e9;
        border-left: 4px solid #00cc66;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.95em;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #0066cc;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f0f7ff;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 5px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0066cc 0%, #00cc66 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 5px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.4);
        transform: translateY(-2px);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border-radius: 10px;
    }
    
    /* Metric styling */
    .metric-box {
        background: linear-gradient(135deg, #0066cc 0%, #00cc66 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-box h3 {
        margin: 0;
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-box p {
        margin: 10px 0 0 0;
        font-size: 2em;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models and Config
# -----------------------------
@st.cache_resource
def load_models():
    # Resolve artifacts directory relative to this file (project root/artifacts)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")

    # Load XGBoost
    xgb = XGBRegressor()
    xgb_path = os.path.join(artifacts_dir, "xgboost_model_20251122_103557.json")
    xgb.load_model(xgb_path)

    # Load CatBoost
    cat = CatBoostRegressor()
    cat_path = os.path.join(artifacts_dir, "catboost_model_20251122_103557.cbm")
    cat.load_model(cat_path)

    # Load feature columns
    features_path = os.path.join(artifacts_dir, "feature_columns_20251122_103557.pkl")
    with open(features_path, "rb") as f:
        feature_columns = pickle.load(f)

    # Load ensemble config
    ensemble_path = os.path.join(artifacts_dir, "ensemble_config_20251122_103557.json")
    with open(ensemble_path, "r") as f:
        ensemble_config = json.load(f)

    return xgb, cat, feature_columns, ensemble_config


xgb_model, cat_model, feature_columns, ensemble_config = load_models()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(df):
    # Make a copy
    df = df.copy()
    
    # Check if any feature is missing
    missing = [c for c in feature_columns if c not in df.columns]
    
    # Attempt to compute features from 'date' and 'sales'
    if missing:
        # Check if we can derive features
        has_date = "date" in df.columns
        has_sales = "sales" in df.columns
        
        if has_date and has_sales:
            try:
                # Convert date to datetime
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                
                # Compute lags
                for lag in [1, 7, 14, 30]:
                    df[f"sales_lag_{lag}"] = df["sales"].shift(lag)
                
                # Compute rolling mean and std
                for window in [7, 14, 30]:
                    df[f"sales_rolling_mean_{window}"] = df["sales"].rolling(window).mean()
                    df[f"sales_rolling_std_{window}"] = df["sales"].rolling(window).std()
                
                # Re-check missing
                missing = [c for c in feature_columns if c not in df.columns]
            except Exception as e:
                raise ValueError(f"Error computing features: {str(e)}")
        else:
            missing_cols = []
            if not has_date:
                missing_cols.append("'date'")
            if not has_sales:
                missing_cols.append("'sales'")
            raise ValueError(
                f"Missing required columns: {', '.join(missing_cols)}. "
                f"Unable to compute missing features: {missing}"
            )
    
    # Fill any remaining missing features with 0
    for col in missing:
        df[col] = 0

    # Keep only required features
    df_features = df[feature_columns].fillna(0)

    # Predict with both models
    xgb_pred = xgb_model.predict(df_features)
    cat_pred = cat_model.predict(df_features)

    # Weighted ensemble
    weights = ensemble_config.get("weights", {"xgb": 0.5, "cat": 0.5})
    final_pred = weights["xgb"] * xgb_pred + weights["cat"] * cat_pred

    return final_pred


# ========================================
# MAIN UI LAYOUT
# ========================================

# Header with sidebar toggle
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    if st.button("☰", help="Toggle Sidebar", key="sidebar_toggle"):
        st.session_state.sidebar_open = not st.session_state.get("sidebar_open", True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>📊 Sales Forecasting Dashboard</h1>
        <p>Predict future sales with machine learning precision</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.markdown("### ⚙️ Model Information")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("XGBoost Weight", f"{ensemble_config.get('weights', {}).get('xgb', 0.5):.2%}")
        with col2:
            st.metric("CatBoost Weight", f"{ensemble_config.get('weights', {}).get('cat', 0.5):.2%}")
    
    st.divider()
    st.markdown("""
    ### 📋 Instructions
    1. Prepare your CSV file with sales data
    2. Include 'date' and 'sales' columns
    3. Upload the file below
    4. Get instant predictions!
    """)
    
    st.divider()
    st.markdown("""
    ### 🔧 Requirements
    - **date**: Date column (any format)
    - **sales**: Historical sales values
    - Optional: Feature columns for direct input
    """)

# Main content area
tab1, tab2 = st.tabs(["📤 Upload & Predict", "📖 Help"])

with tab1:
    # Upload section
    st.markdown('<div class="section-header">📤 Upload Your Data</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here",
        type=["csv"],
        help="File should contain 'date' and 'sales' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data
            st.markdown('<div class="section-header">📊 Data Preview</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Total Records</h3>
                    <p>{len(df):,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Total Columns</h3>
                    <p>{len(df.columns)}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                if "sales" in df.columns:
                    avg_sales = df["sales"].mean()
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Avg Sales</h3>
                        <p>${avg_sales:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("👀 View Full Dataset", expanded=True):
                st.dataframe(df, use_container_width=True, height=300)
            
            # Show column information
            with st.expander("📋 Column Information", expanded=True):
                st.write("**Available Columns:**")
                col_info = []
                for col in df.columns:
                    col_info.append(f"- `{col}` ({df[col].dtype})")
                st.write("\n".join(col_info))
                
                # Check for common column name variations
                st.write("\n**Column Mapping Helper:**")
                possible_date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year'])]
                possible_sales_cols = [col for col in df.columns if any(term in col.lower() for term in ['sales', 'revenue', 'amount', 'value', 'price', 'total'])]
                
                if possible_date_cols:
                    st.info(f"📅 Possible date columns: `{', '.join(possible_date_cols)}`")
                if possible_sales_cols:
                    st.info(f"💰 Possible sales columns: `{', '.join(possible_sales_cols)}`")
                
                # Column renaming option
                st.write("\n**Need to rename columns?**")
                col_rename_a, col_rename_b = st.columns(2)
                with col_rename_a:
                    date_col = st.selectbox("Select date column:", df.columns, key="date_select")
                with col_rename_b:
                    sales_col = st.selectbox("Select sales column:", df.columns, key="sales_select")
                
                # Rename if different from default
                if date_col != "date" or sales_col != "sales":
                    st.info("ℹ️ Columns will be automatically renamed during prediction")
                    # Create a mapping for later use
                    st.session_state.column_mapping = {
                        date_col: "date",
                        sales_col: "sales"
                    }
                else:
                    st.session_state.column_mapping = {}
            
            # Prediction section
            st.markdown('<div class="section-header">🔮 Generate Predictions</div>', unsafe_allow_html=True)
            
            # Process predictions
            if st.button("🚀 Predict Sales", use_container_width=True, type="primary"):
                with st.spinner("🔄 Processing predictions..."):
                    try:
                        # Apply column mapping if needed
                        df_processed = df.copy()
                        if hasattr(st.session_state, 'column_mapping') and st.session_state.column_mapping:
                            df_processed = df_processed.rename(columns=st.session_state.column_mapping)
                        
                        predictions = predict(df_processed)
                        df["Predicted_Sales"] = predictions
                        
                        st.markdown("""
                        <div class="success-message">
                            ✅ <strong>Prediction completed successfully!</strong> Your forecasts are ready.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Results section
                        st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Predictions Count",
                                len(predictions),
                                delta=None
                            )
                        with col2:
                            st.metric(
                                "Avg Predicted Sales",
                                f"${predictions.mean():,.0f}",
                                delta=None
                            )
                        with col3:
                            st.metric(
                                "Max Prediction",
                                f"${predictions.max():,.0f}",
                                delta=None
                            )
                        with col4:
                            st.metric(
                                "Min Prediction",
                                f"${predictions.min():,.0f}",
                                delta=None
                            )
                        
                        # Display results table
                        with st.expander("📊 View Detailed Results", expanded=True):
                            st.dataframe(df, use_container_width=True, height=400)
                        
                        # Download section
                        st.markdown('<div class="section-header">⬇️ Export Results</div>', unsafe_allow_html=True)
                        
                        @st.cache_data
                        def convert_df(df):
                            return df.to_csv(index=False).encode("utf-8")

                        csv = convert_df(df)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="sales_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as pred_error:
                        st.error(f"❌ Prediction Error: {str(pred_error)}")
                        st.info("Please check your column mapping above and try again.")

        except ValueError as e:
            st.error(f"⚠️ Data Error: {e}")
            st.warning("Please ensure your CSV contains 'date' and 'sales' columns with proper data types.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing your file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains the required columns.")
    else:
        st.info(
            "👆 Upload a CSV file to get started. Your file should contain 'date' and 'sales' columns.",
            icon="ℹ️"
        )

with tab2:
    st.markdown("""
    ### 📖 Help & Documentation
    
    #### 📋 Required Columns
    - **date**: Date column (any recognizable date format)
    - **sales**: Historical sales values (numeric)
    
    #### 🔧 Optional Columns
    The model can automatically compute:
    - Lag features (1, 7, 14, 30 days)
    - Rolling mean (7, 14, 30 days)
    - Rolling standard deviation (7, 14, 30 days)
    
    #### 🤖 Model Details
    This dashboard uses an ensemble approach combining:
    - **XGBoost**: Gradient boosting machine learning model
    - **CatBoost**: Categorical boosting model
    - Weighted ensemble for optimal predictions
    
    #### 📊 Data Format Example
    ```
    date,sales
    2024-01-01,1000
    2024-01-02,1200
    2024-01-03,1100
    ```
    
    #### ⚠️ Troubleshooting
    - **Missing features error**: Ensure 'date' and 'sales' columns exist
    - **File format error**: Use CSV format (.csv)
    - **Prediction errors**: Check that sales values are numeric
    
    #### 💡 Tips
    - Ensure dates are in chronological order for better lag feature computation
    - Use at least 30 historical data points for reliable predictions
    - Seasonal patterns will be captured by rolling window features
    """)
    
    with st.expander("🎓 Machine Learning Models Used"):
        st.markdown("""
        **XGBoost (eXtreme Gradient Boosting)**
        - Fast and efficient gradient boosting framework
        - Handles non-linear relationships in sales data
        - Weight in ensemble: {:.2%}
        
        **CatBoost (Categorical Boosting)**
        - Specialized for handling categorical features
        - Robust against overfitting
        - Weight in ensemble: {:.2%}
        """.format(
            ensemble_config.get('weights', {}).get('xgb', 0.5),
            ensemble_config.get('weights', {}).get('cat', 0.5)
        ))

