import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime
from pathlib import Path


# =========================================================
# Page Configuration & Styling
# =========================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
        padding-bottom: 1rem;
    }
    h2, h3 {
        color: #1f77b4;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .metric-card h3 {
        margin-top: 0;
        color: #1f77b4;
    }
    .metric-card p {
        margin-bottom: 0;
        color: #6b7280;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# Sidebar Navigation
# =========================================================

st.sidebar.title("📊 Sales Forecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📚 Feature Guide", "🔮 Make Predictions", "📈 Batch Forecast", "ℹ️ About"]
)

# =========================================================
# Load Keras Model
# =========================================================
@st.cache_resource
def load_keras_model():
    try:
        # Model is in the same directory as this script
        model_path = Path(__file__).parent / "model_nn.keras"
        model = load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_keras_model()

# =========================================================
# Feature Vector Creator (103 features)
# =========================================================
def create_feature_vector(store_nbr, family, date, onpromotion, dcoilwtico,
                          is_holiday, is_event, is_work_day, is_earthquake):
    features = np.zeros(103, dtype=np.float32)

    # ---- Convert & normalize date ----
    date_obj = pd.to_datetime(date)
    features[0] = float(onpromotion)
    features[1] = float(dcoilwtico)
    features[2] = float(is_holiday)
    features[3] = float(is_event)
    features[4] = float(is_work_day)
    features[5] = float(is_earthquake)
    features[6] = date_obj.day / 31.0
    features[7] = date_obj.month / 12.0
    features[8] = 1.0  # fixed normalized year

    # ---- Store one-hot: index 9–62 ----
    store_idx = 9 + (store_nbr - 1)
    if 9 <= store_idx <= 62:
        features[store_idx] = 1.0

    # ---- Family one-hot: index 63–95 ----
    family_to_index = {
        "AUTOMOTIVE": 63, "BABY CARE": 64, "BEAUTY": 65, "BEVERAGES": 66,
        "BOOKS": 67, "BREAD/BAKERY": 68, "CELEBRATION": 69, "CLEANING": 70,
        "DAIRY": 71, "DELI": 72, "EGGS": 73, "FROZEN FOODS": 74,
        "GROCERY I": 75, "GROCERY II": 76, "HARDWARE": 77,
        "HOME AND KITCHEN I": 78, "HOME AND KITCHEN II": 79,
        "HOME APPLIANCES": 80, "HOME CARE": 81, "LADIESWEAR": 82,
        "LAWN AND GARDEN": 83, "LINGERIE": 84, "LIQUOR,WINE,BEER": 85,
        "MAGAZINES": 86, "MEATS": 87, "PERSONAL CARE": 88,
        "PET SUPPLIES": 89, "PLAYERS AND ELECTRONICS": 90, "POULTRY": 91,
        "PREPARED FOODS": 92, "PRODUCE": 93, "SCHOOL AND OFFICE SUPPLIES": 94,
        "SEAFOOD": 95
    }

    if family in family_to_index:
        features[family_to_index[family]] = 1.0

    # ---- Day of week one-hot: index 96–102 ----
    day_to_index = {
        "Friday": 96, "Monday": 97, "Saturday": 98,
        "Sunday": 99, "Thursday": 100, "Tuesday": 101, "Wednesday": 102
    }

    day_name = date_obj.day_name()
    if day_name in day_to_index:
        features[day_to_index[day_name]] = 1.0

    return features.reshape(1, -1)

# =========================================================
# Predict Function
# =========================================================
def make_prediction(model, **kwargs):
    features = create_feature_vector(**kwargs)
    prediction = model.predict(features, verbose=0)[0][0]
    return float(max(0, prediction))


# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":
    st.title("🏠 Sales Forecasting Dashboard")
    st.markdown("### Welcome to the Sales Demand Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>ML-powered forecasting with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Results</h3>
            <p>Get instant predictions for your sales data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Interactive Visualizations</h3>
            <p>Explore trends and patterns in your forecasts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. *📚 Feature Guide* - Learn about the key features used in the model
    2. *🔮 Make Predictions* - Enter sales data manually for single predictions
    3. *📈 Batch Forecast* - Upload CSV file for multiple predictions
    4. *ℹ️ About* - Learn more about the project
    """)


# ============================================================================ 
# FEATURE GUIDE PAGE
# ============================================================================

elif page == "📚 Feature Guide":
    st.title("📚 Feature Guide for Sales Forecasting")
    st.markdown("### Understanding the Key Features Used in the Model")
    
    # Main categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏪 Store Features", 
        "📦 Product Features", 
        "💰 Economic Factors", 
        "📅 Time Features",
        "🔧 Promotional Features"
    ])
    
    # Tab 1: Store Features
    with tab1:
        st.markdown("## 🏪 Store Number")
        st.info("""
        *Range:* 1 to 54  
        *Type:* Categorical / Numeric
        
        *Description:*  
        Unique identifier for each store location.  
        Reflects differences in:  
        - Store size and layout  
        - Customer demographics  
        - Sales patterns
        
        *Impact on Sales:*  
        High: Different stores can have large variations in daily sales.
        
        *Example:*  
        Store #44 (capital city) may sell 3x more than Store #12 (small town)
        """)
    
    # Tab 2: Product Features
    with tab2:
        st.markdown("## 📦 Product Family")
        st.info("""
        *Count:* 33 product categories  
        *Type:* Categorical
        
        *Major Categories:*  
        - Food & Beverages: GROCERY I & II, BEVERAGES, PRODUCE, DAIRY, BREAD/BAKERY, MEATS, POULTRY, SEAFOOD, DELI, FROZEN FOODS  
        - Home & Personal Care: CLEANING, HOME CARE, PERSONAL CARE, BEAUTY, BABY CARE, PET SUPPLIES  
        - Other Categories: AUTOMOTIVE, HARDWARE, HOME APPLIANCES, BOOKS & MAGAZINES, CELEBRATION, SCHOOL SUPPLIES
        
        *Impact on Sales:*  
        Sales patterns and seasonality differ across families.  
        GROCERY I alone may account for 15-20% of total sales.
        """)
        
    # Tab 3: Economic Factors
    with tab3:
        st.markdown("## 💰 Oil Price (dcoilwtico)")
        st.info(""" 
        *Type:* Numeric (Continuous)  
        
        *Why It Matters:*  
        Ecuador is an oil-dependent economy:  
        - Oil exports affect government income  
        - Higher oil price → more consumer spending  
        - Affects transportation & retail costs
        
        *Impact on Sales:*  
        Medium-High: 10-20% correlation with sales
        """)
    
    # Tab 4: Time Features
    with tab4:
        st.markdown("## 📅 Time-Based Features")
        st.info("""
        *Type:* DateTime / Categorical / Numeric  
        *Features Extracted From Date:*  
        - Year, Month, Day, Day of Week  
        - Is weekend? Is month start/end?  
        
        *Impact on Sales:*  
        Very High: Captures weekly and seasonal patterns
        
        *Examples:*  
        - Fridays → higher sales  
        - December → peak sales due to holidays
        """)
    
    # Tab 5: Promotional Features
    with tab5:
        st.markdown("## 🔧 Promotional Features")
        st.info("""
        *Type:* Numeric
        *Examples:*  
        
        *Impact on Sales:*  
        High: Promotions can increase sales by 20-150% depending on product
        
        *Examples:*  
        - Food items: +30-50% with promotion  
        - Non-essential items: +50-100%  
        - Combination promotions: +80-150%
        """)

# =========================================================
# MAKE PREDICTION PAGE
# =========================================================
elif page == "🔮 Make Predictions":

    st.title("🔮 Predict Sales")

    col1, col2 = st.columns(2)

    with col1:
        store_nbr = st.number_input("Store Number", min_value=1, max_value=54, value=1)
        family = st.selectbox("Product Family", [
            "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
            "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
            "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
            "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
            "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
            "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
            "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
            "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES",
            "SEAFOOD"
        ])
        date = st.date_input("Date")

    with col2:
        onpromotion = st.number_input("On Promotion", min_value=0.0, max_value=1.0, value=0.0)
        dcoilwtico = st.number_input("Oil Price (DCOILWTICO)", value=0.25)
        is_holiday = st.checkbox("Holiday")
        is_event = st.checkbox("Event")
        is_work_day = st.checkbox("Work Day", value=True)
        is_earthquake = st.checkbox("Earthquake")

    if st.button("Predict 🔮", use_container_width=True):
        if model is None:
            st.error("Model not loaded.")
        else:
            pred = make_prediction(
                model,
                store_nbr=store_nbr,
                family=family,
                date=str(date),
                onpromotion=onpromotion,
                dcoilwtico=dcoilwtico,
                is_holiday=is_holiday,
                is_event=is_event,
                is_work_day=is_work_day,
                is_earthquake=is_earthquake
            )
            st.success(f"📈 **Predicted Sales:** {pred:.2f}")

# =========================================================
# BATCH FORECAST PAGE
# =========================================================
elif page == "📈 Batch Forecast":
    st.title("📦 Batch Forecast")
    st.markdown("### Upload your CSV file to predict sales using AI-powered forecasting")
    
    if model is None:
        st.warning("⚠️ **Please wait for the model to load!**")
        st.info("The model is being loaded automatically. Please refresh if it doesn't load.")
    else:
        st.markdown("---")
        
        # Feature explanation
        with st.expander("ℹ️ Understanding CSV Requirements", expanded=False):
            st.markdown("""
            ### 📋 Supported CSV Formats:
            
            **Option 1: Raw Input Columns** (Recommended)
            Your CSV file should contain the following columns:
            - **store_nbr**: Store number (1-54)
            - **family**: Product family (e.g., GROCERY I, BEVERAGES, DAIRY)
            - **date**: Date in format YYYY-MM-DD
            - **onpromotion**: Promotion flag (0.0 or 1.0)
            - **dcoilwtico**: Oil price value
            - **is_holiday**: Holiday flag (0 or 1, or True/False)
            - **is_event**: Event flag (0 or 1, or True/False)
            - **is_work_day**: Work day flag (0 or 1, or True/False)
            - **is_earthquake**: Earthquake flag (0 or 1, or True/False)
            
            **Option 2: Preprocessed Features**
            Your CSV file can contain exactly 103 feature columns (already processed).
            The system will detect this format automatically.
            
            ---
            
            ### 🧠 How It Works:
            
            - **Raw Input**: The system converts each row into a 103-feature vector using feature engineering
            - **Preprocessed**: The system uses the features directly for prediction
            """)
        
        # File uploader
        st.markdown("### 📁 Upload Your Data")
        uploaded = st.file_uploader(
            "Choose CSV file with input features",
            type=['csv'],
            help="Upload a CSV file containing the required columns for prediction."
        )

        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                result_df = df.copy()
                
                # Get model expected features
                model_features = model.input_shape[-1] if hasattr(model, 'input_shape') else 103
                
                # Detect CSV format
                required_cols = ["store_nbr", "family", "date", "onpromotion", 
                               "dcoilwtico", "is_holiday", "is_event", 
                               "is_work_day", "is_earthquake"]
                has_raw_cols = all(col in df.columns for col in required_cols)
                has_processed_features = len(df.columns) == model_features
                
                # Data preview and validation
                st.markdown("### 📋 Data Preview & Validation")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Uploaded Data Preview**")
                    st.dataframe(df.head(100), use_container_width=True, height=400)
                
                with col2:
                    st.markdown("**📊 Data Statistics**")
                    st.metric("Total Rows", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))
                    st.metric("Expected Features", model_features)
                    
                    st.markdown("---")
                    # Format detection
                    if has_processed_features:
                        st.success("✅ **Format: Preprocessed Features**")
                        st.info(f"Detected {model_features} feature columns. Will use directly.")
                    elif has_raw_cols:
                        st.success("✅ **Format: Raw Input Columns**")
                        st.info("Will convert to feature vectors automatically.")
                    else:
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        st.error(f"❌ **Format not recognized**")
                        st.warning(f"Missing columns: {', '.join(missing_cols)}")
                        st.info(f"Expected either:\n- Raw columns: {', '.join(required_cols)}\n- Or {model_features} processed feature columns")
                
                st.markdown("---")
                
                # Prediction button
                if st.button("🚀 Predict Sales", use_container_width=True, type="primary"):
                    if not (has_raw_cols or has_processed_features):
                        st.error("❌ CSV format not recognized. Please check the requirements above.")
                    else:
                        with st.spinner("🔮 Predicting sales..."):
                            predictions = []
                            progress_bar = st.progress(0)
                            
                            try:
                                if has_processed_features:
                                    # Direct prediction with processed features
                                    feature_data = df.values.astype(np.float32)
                                    batch_predictions = model.predict(feature_data, verbose=0)
                                    predictions = [max(0, float(pred[0])) for pred in batch_predictions]
                                    progress_bar.progress(1.0)
                                else:
                                    # Convert raw columns to features
                                    for idx, (_, row) in enumerate(df.iterrows()):
                                        try:
                                            # Convert boolean/string values to proper format
                                            onpromotion = float(row["onpromotion"])
                                            dcoilwtico = float(row["dcoilwtico"])
                                            is_holiday = float(bool(row["is_holiday"]))
                                            is_event = float(bool(row["is_event"]))
                                            is_work_day = float(bool(row["is_work_day"]))
                                            is_earthquake = float(bool(row["is_earthquake"]))
                                            
                                            features = create_feature_vector(
                                                store_nbr=int(row["store_nbr"]),
                                                family=str(row["family"]),
                                                date=str(row["date"]),
                                                onpromotion=onpromotion,
                                                dcoilwtico=dcoilwtico,
                                                is_holiday=is_holiday,
                                                is_event=is_event,
                                                is_work_day=is_work_day,
                                                is_earthquake=is_earthquake
                                            )
                                            pred = model.predict(features, verbose=0)[0][0]
                                            predictions.append(max(0, float(pred)))
                                            
                                            # Update progress
                                            progress_bar.progress((idx + 1) / len(df))
                                        except Exception as e:
                                            st.warning(f"⚠️ Error processing row {idx + 1}: {str(e)}")
                                            predictions.append(0.0)
                                
                                progress_bar.empty()
                                result_df["predicted_sales"] = predictions
                                
                                # Display results
                                st.markdown("### 📊 Prediction Results")
                                st.dataframe(result_df.head(100), use_container_width=True, height=400)
                                
                                # Statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Predictions", len(predictions))
                                with col2:
                                    st.metric("Mean Predicted Sales", f"{np.mean(predictions):.2f}")
                                with col3:
                                    st.metric("Total Predicted Sales", f"{np.sum(predictions):,.2f}")
                                
                                # Download section
                                st.markdown("---")
                                st.markdown("#### 💾 Download Predictions")
                                csv_data = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                st.success("✅ Predictions completed successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Error during prediction: {str(e)}")
                                st.exception(e)
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.exception(e)

# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ### 📊 Sales Forecasting and Demand Prediction System
    
    This application uses Machine Learning to predict future sales and demand based on historical data,
    helping businesses optimize inventory management, staffing, and marketing strategies.
    
    ---
    
    ### 🎯 Key Features
    
    - *Accurate Predictions*: ML-powered forecasting with multiple algorithms
    - *Real-time Forecasting*: Get instant predictions for your sales data  
    - *Batch Processing*: Upload CSV files for multiple predictions
    
    ---
    
    ### 📝 How to Use
    
    1. Navigate to *Make Predictions* for single forecasts
    2. Use *Batch Forecast* to upload CSV files
    """)
    
   
    st.markdown("---")
    
    # About Team
    st.markdown("""
    ## Our Team 
    - *Abdelrahman Saeed* - [LinkedIn](https://www.linkedin.com/in/abdelrahman-abdelraouf004/)
    - *Abdelrahman Youssry* - [LinkedIn](https://www.linkedin.com/in/abdelrahman-yousry-271816269/)
    - *Farida Sabra* - [LinkedIn](https://www.linkedin.com/in/farida-sabra)
    - *Hossam Eldin Mahmod* - [LinkedIn](http://linkedin.com/in/hossam-eldin-m-hmady)
    - *Rana Mohammed* - [LinkedIn](https://www.linkedin.com/in/rana-mohammed1)
    - *Sara Basheer* - [LinkedIn](http://linkedin.com/in/sara-basheer)
    """)