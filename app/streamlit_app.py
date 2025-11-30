import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
        padding-bottom: 1rem;
    }
    h2 {
        color: #1f77b4;
        font-weight: 600;
    }
    h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        margin: 10px 0;
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
    .info-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .feature-explanation {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)


class SalesPredictionApp:
    """Sales prediction application using neural network model"""
    
    def __init__(self):
        # Define paths
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Model path
        self.model_path = self.models_dir / "model_nn.keras"
        
        # Initialize session state
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'scalers' not in st.session_state:
            st.session_state.scalers = {}
        if 'last_prediction_df' not in st.session_state:
            st.session_state.last_prediction_df = None
    
    def load_model(self):
        """Load the Keras model"""
        try:
            if not self.model_path.exists():
                st.error(f"❌ Model not found at: {self.model_path}")
                return False
            
            st.session_state.model = keras.models.load_model(self.model_path)
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def get_available_scalers(self):
        """Get list of available scaler files"""
        if not self.artifacts_dir.exists():
            return {}
        
        scalers = {}
        for f in self.artifacts_dir.glob("scaler_*.pkl"):
            scalers[f.name] = f
        return scalers
    
    def load_all_scalers(self):
        """Load all available scalers"""
        available_scalers = self.get_available_scalers()
        loaded_count = 0
        
        for name, path in available_scalers.items():
            loaded = False
            last_error = None
            
            # Method 1: Try joblib first (scikit-learn scalers are often saved with joblib)
            try:
                scaler = joblib.load(path)
                st.session_state.scalers[name] = scaler
                loaded_count += 1
                loaded = True
            except Exception as e1:
                last_error = f"joblib: {str(e1)}"
                
                # Method 2: Try pickle with default protocol
                try:
                    with open(path, 'rb') as f:
                        scaler = pickle.load(f)
                    st.session_state.scalers[name] = scaler
                    loaded_count += 1
                    loaded = True
                except Exception as e2:
                    last_error = f"pickle: {str(e2)}"
                    
                    # Method 3: Try pickle with latin1 encoding (for Python 2/3 compatibility)
                    try:
                        with open(path, 'rb') as f:
                            scaler = pickle.load(f, encoding='latin1')
                        st.session_state.scalers[name] = scaler
                        loaded_count += 1
                        loaded = True
                    except Exception as e3:
                        last_error = f"pickle(latin1): {str(e3)}"
            
            if not loaded:
                failed_scalers.append((name, last_error))
        
        # Report failures if any
        if failed_scalers:
            for name, error in failed_scalers:
                st.warning(f"⚠️ Could not load {name}: {error}")
        
        return loaded_count
    
    def predict_sales(self, df, use_scaling=True):
        """Make sales predictions on the dataframe"""
        try:
            # Get the feature data
            feature_data = df.values
            
            # Apply scaling if requested and scalers are loaded
            if use_scaling and st.session_state.scalers:
                # This is a simplified version - you may need to apply different scalers to different columns
                # For now, we'll use the first available scaler
                scaler_name = list(st.session_state.scalers.keys())[0]
                scaler = st.session_state.scalers[scaler_name]
                feature_data = scaler.transform(feature_data)
            
            # Make predictions
            predictions = st.session_state.model.predict(feature_data, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            raise
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            # Enhanced Header
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #1f77b4; margin-bottom: 0.5rem;'>📊 Sales Forecasting</h1>
                <p style='color: #6b7280; font-size: 0.9rem;'>AI-Powered Prediction System</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            # Model Loading with enhanced display
            st.markdown("### 🤖 Model Status")
            status = "✅ Loaded" if st.session_state.model_loaded else "❌ Not Loaded"
            status_color = "#10b981" if st.session_state.model_loaded else "#ef4444"
            st.markdown(f"""
            <div style='padding: 0.75rem; background-color: #f0f2f6; border-radius: 0.5rem; border-left: 4px solid {status_color};'>
                <strong>Status:</strong> {status}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Load Model", use_container_width=True):
                with st.spinner("🔄 Loading model..."):
                    if self.load_model():
                        st.success("✅ Model loaded successfully!")
                        model = st.session_state.model
                        st.markdown(f"""
                        <div style='padding: 0.75rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                            <strong>Input Features:</strong> {model.input_shape[-1]}<br>
                            <strong>Parameters:</strong> {model.count_params():,}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Scaler Loading with enhanced display
            st.markdown("### 📊 Feature Scalers")
            available_scalers = self.get_available_scalers()
            
            if available_scalers:
                st.markdown(f"""
                <div style='padding: 0.75rem; background-color: #eff6ff; border-radius: 0.5rem; border-left: 4px solid #3b82f6;'>
                    Found <strong>{len(available_scalers)}</strong> scaler(s)
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📥 Load All Scalers", use_container_width=True):
                    with st.spinner("Loading scalers..."):
                        count = self.load_all_scalers()
                        if count > 0:
                            st.success(f"✅ Loaded {count} scaler(s)!")
                            with st.expander("View loaded scalers", expanded=False):
                                for name in st.session_state.scalers.keys():
                                    st.caption(f"• {name}")
                
                if st.session_state.scalers:
                    st.markdown(f"""
                    <div style='padding: 0.75rem; background-color: #d1fae5; border-radius: 0.5rem; border-left: 4px solid #10b981;'>
                        ✅ <strong>{len(st.session_state.scalers)}</strong> scaler(s) active
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No scalers found in artifacts/")
            
            st.markdown("---")
            
            # Enhanced Info Section
            st.markdown("### ℹ️ How to Use")
            with st.expander("📖 Quick Start Guide", expanded=False):
                st.markdown("""
                **Step-by-Step:**
                1. **Load Model** - Click the "Load Model" button above
                2. **Load Scalers** (Optional) - Load feature scalers for better accuracy
                3. **Upload CSV** - Upload your CSV file with all required features
                4. **Configure** - Set prediction options (scaling, etc.)
                5. **Predict** - Click "Predict Sales" to generate forecasts
                6. **Download** - Download results with IDs and predicted sales
                
                **CSV Requirements:**
                - ✅ Must contain all required features
                - ✅ No missing values
                - ✅ Correct column order
                - ✅ Proper data types
                """)
            
            st.markdown("---")
            
            # Statistics
            if st.session_state.prediction_history:
                st.subheader("📊 Session Stats")
                st.metric("Total Predictions", len(st.session_state.prediction_history))
            
            st.markdown("---")
            st.caption("v1.0 | Sales Forecasting System")
    
    def render_main_content(self):
        """Render main prediction interface"""
        st.header("🔮 Sales Prediction System")
        st.markdown("### Upload your CSV file to predict sales using AI-powered forecasting")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ **Please load the model from the sidebar first!**")
            st.info("👈 Click 'Load Model' in the sidebar to get started")
            
            # Show feature cards
            st.markdown("---")
            st.markdown("### 🎯 Key Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>🎯 Accurate Predictions</h3>
                    <p>Neural network-powered forecasting with high accuracy</p>
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
                    <h3>📊 Batch Processing</h3>
                    <p>Process multiple records efficiently</p>
                </div>
                """, unsafe_allow_html=True)
            
            return
        
        st.markdown("---")
        
        # Feature explanation section
        with st.expander("ℹ️ Understanding Sales Prediction Features", expanded=False):
            st.markdown("""
            ### 📋 Feature Descriptions
            
            **Basic Information:**
            - **Store Number** (1-54): Unique identifier for each store location
            - **Date**: The date for which you want to predict sales
            - **Product Family**: Category of products (e.g., GROCERY, BEVERAGES, DAIRY)
            
            **Store Characteristics:**
            - **Store Type** (A-E): Classification of store size and format
              - Type A: Largest stores
              - Type E: Smallest stores
            - **Store Cluster** (1-17): Grouping of similar stores based on characteristics
            
            **Economic Factors:**
            - **Oil Price**: Daily oil price in USD
              - Ecuador's economy is oil-dependent
              - Higher oil prices typically mean higher consumer spending
              - Affects transportation costs and consumer behavior
            
            **Sales Indicators:**
            - **Items on Promotion**: Number of items from this product family currently on promotion
              - Promotions typically increase sales by 20-40%
              - Important for demand forecasting
            
            - **Transactions**: Total number of customer transactions at the store that day
              - Higher transactions usually mean higher sales
              - Indicates store traffic and customer activity
            
            **Special Events:**
            - **Is Holiday?**: Whether the date is a national or local holiday
              - Holidays significantly impact sales patterns
              - Some product families sell more (celebrations), others less (offices closed)
            
            ---
            
            ### 🧠 How the Model Uses These Features:
            
            The neural network model combines these inputs with **engineered features** like:
            - **Historical patterns**: Sales from previous weeks/months
            - **Seasonality**: Day of week, month, quarter effects
            - **Trends**: Overall sales trends over time
            - **Interactions**: How promotions work differently on holidays
            - **Rolling averages**: Recent sales performance
            
            All these factors together help predict future sales accurately! 📊
            """)
        
        # File uploader with better styling
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file with features",
            type=['csv'],
            help="Upload a CSV file containing the feature columns needed for prediction. The file should have all required features in the correct order."
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                original_df = df.copy()  # Keep original for final output
                
                # Display data info with enhanced styling
                st.markdown("### 📋 Data Preview & Validation")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Uploaded Data Preview**")
                    st.dataframe(df.head(20), use_container_width=True, height=300)
                
                with col2:
                    st.markdown("**📊 Data Statistics**")
                    st.metric("Total Rows", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))
                    
                    model_features = st.session_state.model.input_shape[-1]
                    st.metric("Required Features", model_features)
                    
                    st.markdown("---")
                    if len(df.columns) == model_features:
                        st.success("✅ **Column count matches!**")
                    else:
                        st.error(f"❌ **Mismatch!** Expected {model_features}, got {len(df.columns)}")
                
                st.markdown("---")
                
                # Prediction options with better layout
                st.markdown("### ⚙️ Prediction Configuration")
                
                col_a, col_b, col_c = st.columns([2, 1, 1])
                
                with col_a:
                    st.markdown("**Configure your prediction settings**")
                    use_scaling = st.checkbox(
                        "Apply Feature Scaling",
                        value=len(st.session_state.scalers) > 0,
                        disabled=len(st.session_state.scalers) == 0,
                        help="Apply loaded scalers to normalize features. This improves prediction accuracy when scalers are available."
                    )
                    if len(st.session_state.scalers) == 0:
                        st.caption("ℹ️ No scalers loaded. Scaling disabled.")
                
                with col_b:
                    st.markdown("**Ready to predict?**")
                    st.markdown("")  # Spacing
                
                with col_c:
                    st.markdown("")  # Spacing
                    st.markdown("")  # Spacing
                    if st.button("🚀 Predict Sales", use_container_width=True, type="primary"):
                        # Validate data shape
                        if len(df.columns) != model_features:
                            st.error(f"❌ Column mismatch! Expected {model_features} columns, got {len(df.columns)}")
                        else:
                            with st.spinner("🔮 Predicting sales..."):
                                try:
                                    # Make predictions
                                    predictions = self.predict_sales(df, use_scaling)
                                    
                                    # Add predictions to original dataframe
                                    result_df = original_df.copy()
                                    result_df['predicted_sales'] = predictions
                                    
                                    # Store in session state
                                    st.session_state.last_prediction_df = result_df
                                    
                                    # Add to history
                                    st.session_state.prediction_history.append({
                                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rows': len(result_df),
                                        'mean_sales': predictions.mean(),
                                        'total_sales': predictions.sum()
                                    })
                                    
                                    st.success("✅ Predictions completed successfully!")
                                    
                                except Exception as e:
                                    st.error(f"Error during prediction: {str(e)}")
                
                # Display results if available
                if st.session_state.last_prediction_df is not None:
                    st.markdown("---")
                    st.markdown("### 📈 Prediction Results")
                    st.success("✅ Predictions completed successfully! Review the results below.")
                    
                    result_df = st.session_state.last_prediction_df
                    predictions = result_df['predicted_sales'].values
                    
                    # Enhanced Statistics with better presentation
                    st.markdown("#### 📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Total Predicted Sales", 
                            f"{predictions.sum():,.2f}",
                            help="Sum of all predicted sales values"
                        )
                    with col2:
                        st.metric(
                            "Average Sales", 
                            f"{predictions.mean():,.2f}",
                            help="Mean of all predicted sales values"
                        )
                    with col3:
                        st.metric(
                            "Min Sales", 
                            f"{predictions.min():,.2f}",
                            help="Minimum predicted sales value"
                        )
                    with col4:
                        st.metric(
                            "Max Sales", 
                            f"{predictions.max():,.2f}",
                            help="Maximum predicted sales value"
                        )
                    
                    st.markdown("---")
                    
                    # Display results table with enhanced styling
                    st.markdown("#### 📊 Complete Results Table")
                    st.dataframe(result_df, use_container_width=True, height=400)
                    
                    st.markdown("---")
                    
                    # Prepare download CSV with id and predicted_sales
                    st.markdown("#### 💾 Download Predictions")
                    st.info("💡 The download file will contain IDs (starting from 3000888) and predicted sales values.")
                    
                    # Create IDs starting from 3000888
                    start_id = 3000888
                    ids = range(start_id, start_id + len(predictions))
                    
                    download_df = pd.DataFrame({
                        'id': ids,
                        'predicted_sales': predictions
                    })
                    
                    # Download button with better styling
                    csv = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV (ID + Sales)",
                        data=csv,
                        file_name=f"sales_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary",
                        help="Download the predictions as a CSV file with ID and predicted_sales columns"
                    )
                    
                    # Show preview of download file
                    with st.expander("👁️ Preview Download File", expanded=False):
                        st.markdown(f"**File Information:**")
                        st.caption(f"• Total rows: {len(download_df):,}")
                        st.caption(f"• ID range: {download_df['id'].min()} to {download_df['id'].max()}")
                        st.caption(f"• Columns: id, predicted_sales")
                        st.markdown("**Preview:**")
                        st.dataframe(download_df.head(20), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.exception(e)
    
    def render_history(self):
        """Render prediction history"""
        st.header("📜 Prediction History")
        st.markdown("### View your past predictions and session statistics")
        
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Summary statistics with enhanced display
            st.markdown("#### 📊 Session Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Predictions", 
                    len(history_df),
                    help="Number of prediction sessions in this history"
                )
            with col2:
                total_rows = history_df['rows'].sum()
                st.metric(
                    "Total Rows Predicted", 
                    f"{total_rows:,}",
                    help="Total number of rows predicted across all sessions"
                )
            with col3:
                avg_sales = history_df['mean_sales'].mean()
                st.metric(
                    "Avg Sales per Session", 
                    f"{avg_sales:,.2f}",
                    help="Average predicted sales per session"
                )
            
            st.markdown("---")
            
            # History table
            st.markdown("#### 📋 Detailed History")
            st.dataframe(history_df, use_container_width=True, height=300)
            
            st.markdown("---")
            
            # Clear history button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                    st.session_state.prediction_history = []
                    st.session_state.last_prediction_df = None
                    st.success("✅ History cleared!")
                    st.rerun()
        else:
            st.markdown("---")
            st.info("""
            📭 **No prediction history yet!**
            
            To see your prediction history:
            1. Go to the "🎯 Predict Sales" tab
            2. Upload a CSV file
            3. Make predictions
            4. Your predictions will appear here
            """)
    
    def run(self):
        """Main application runner"""
        # Enhanced Title Section
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin-bottom: 0.5rem;'>📈 الحج المتنبيء</h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem;'>AI-Powered Sales Prediction Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content tabs with enhanced styling
        tab1, tab2 = st.tabs([
            "🎯 Predict Sales",
            "📜 Prediction History"
        ])
        
        with tab1:
            self.render_main_content()
        
        with tab2:
            self.render_history()
        
        # Enhanced Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6b7280; padding: 20px; background-color: #f9fafb; border-radius: 8px;'>
            <p style='margin: 0; font-size: 0.9rem;'>
                Built with ❤️ using <strong>TensorFlow</strong> & <strong>Streamlit</strong>
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #9ca3af;'>
                Sales Forecasting System v1.0 | Powered by Neural Networks
            </p>
        </div>
        """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    app = SalesPredictionApp()
    app.run()