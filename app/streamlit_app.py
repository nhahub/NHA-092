import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV and cache it."""
    return df.to_csv(index=False).encode('utf-8')


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class SalesPredictionApp:
    """Sales prediction application using neural network model."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.model_path = self.models_dir / "model_nn.keras"
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables."""
        defaults = {
            'model': None,
            'model_loaded': False,
            'prediction_history': [],
            'scalers': {},
            'last_prediction_df': None,
            'last_prediction_csv': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    # ------------------------------------------------------------------------
    # MODEL & SCALER LOADING
    # ------------------------------------------------------------------------
    
    def load_model(self):
        """Load the Keras model."""
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
        """Get list of available scaler files."""
        if not self.artifacts_dir.exists():
            return {}
        return {f.name: f for f in self.artifacts_dir.glob("scaler_*.pkl")}
    
    def load_all_scalers(self):
        """Load all available scalers."""
        available_scalers = self.get_available_scalers()
        loaded_count = 0
        failed_scalers = []
        
        for name, path in available_scalers.items():
            if self._load_single_scaler(name, path):
                loaded_count += 1
            else:
                failed_scalers.append(name)
        
        if failed_scalers:
            for name in failed_scalers:
                st.warning(f"⚠️ Could not load {name}")
        
        return loaded_count
    
    def _load_single_scaler(self, name, path):
        """Attempt to load a single scaler using multiple methods."""
        load_methods = [
            ('joblib', lambda: joblib.load(path)),
            ('pickle', lambda: pickle.load(open(path, 'rb'))),
            ('pickle_latin1', lambda: pickle.load(open(path, 'rb'), encoding='latin1'))
        ]
        
        for method_name, load_func in load_methods:
            try:
                st.session_state.scalers[name] = load_func()
                return True
            except Exception:
                continue
        
        return False
    
    # ------------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------------
    
    def predict_sales(self, df, use_scaling=True):
        """Make sales predictions on the dataframe."""
        try:
            feature_data = df.values
            
            if use_scaling and st.session_state.scalers:
                scaler_name = list(st.session_state.scalers.keys())[0]
                scaler = st.session_state.scalers[scaler_name]
                feature_data = scaler.transform(feature_data)
            
            predictions = st.session_state.model.predict(feature_data, verbose=0)
            return predictions.flatten()
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            raise
    
    # ------------------------------------------------------------------------
    # UI RENDERING - SIDEBAR
    # ------------------------------------------------------------------------
    
    def render_sidebar(self):
        """Render sidebar configuration."""
        with st.sidebar:
            self._render_sidebar_header()
            st.markdown("---")
            
            self._render_model_section()
            st.markdown("---")
            
            self._render_scaler_section()
            st.markdown("---")
            
            self._render_help_section()
            st.markdown("---")
            
            if st.session_state.prediction_history:
                st.subheader("📊 Session Stats")
                st.metric("Total Predictions", len(st.session_state.prediction_history))
            
            st.markdown("---")
            st.caption("v1.0 | Sales Forecasting System")
    
    def _render_sidebar_header(self):
        """Render sidebar header."""
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1f77b4; margin-bottom: 0.5rem;'>📊 Sales Forecasting</h1>
            <p style='color: #6b7280; font-size: 0.9rem;'>AI-Powered Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_model_section(self):
        """Render model loading section."""
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
    
    def _render_scaler_section(self):
        """Render scaler loading section."""
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
    
    def _render_help_section(self):
        """Render help section."""
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
    
    # ------------------------------------------------------------------------
    # UI RENDERING - MAIN CONTENT
    # ------------------------------------------------------------------------
    
    def render_main_content(self):
        """Render main prediction interface."""
        st.header("🔮 Sales Prediction System")
        st.markdown("### Upload your CSV file to predict sales using AI-powered forecasting")
        
        if not st.session_state.model_loaded:
            self._render_model_not_loaded()
            return
        
        st.markdown("---")
        self._render_feature_explanation()
        self._render_file_uploader()
    
    def _render_model_not_loaded(self):
        """Render UI when model is not loaded."""
        st.warning("⚠️ **Please load the model from the sidebar first!**")
        st.info("👈 Click 'Load Model' in the sidebar to get started")
        
        st.markdown("---")
        st.markdown("### 🎯 Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        features = [
            ("🎯 Accurate Predictions", "Neural network-powered forecasting with high accuracy"),
            ("⚡ Real-time Results", "Get instant predictions for your sales data"),
            ("📊 Batch Processing", "Process multiple records efficiently")
        ]
        
        for col, (title, desc) in zip([col1, col2, col3], features):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_feature_explanation(self):
        """Render feature explanation section."""
        with st.expander("ℹ️ Understanding Sales Prediction Features", expanded=False):
            st.markdown("""
            ### 📋 Feature Descriptions
            
            **Basic Information:**
            - **Store Number** (1-54): Unique identifier for each store location
            - **Date**: The date for which you want to predict sales
            - **Product Family**: Category of products (e.g., GROCERY, BEVERAGES, DAIRY)
            
            **Store Characteristics:**
            - **Store Type** (A-E): Classification of store size and format
            - **Store Cluster** (1-17): Grouping of similar stores based on characteristics
            
            **Economic Factors:**
            - **Oil Price**: Daily oil price in USD
            
            **Sales Indicators:**
            - **Items on Promotion**: Number of items from this product family currently on promotion
            - **Transactions**: Total number of customer transactions at the store that day
            
            **Special Events:**
            - **Is Holiday?**: Whether the date is a national or local holiday
            
            ---
            
            ### 🧠 How the Model Uses These Features:
            
            The neural network model combines these inputs with engineered features like historical patterns, 
            seasonality, trends, interactions, and rolling averages to predict future sales accurately! 📊
            """)
    
    def _render_file_uploader(self):
        """Render file uploader and prediction interface."""
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file with features",
            type=['csv'],
            help="Upload a CSV file containing the feature columns needed for prediction."
        )
        
        if uploaded_file is not None:
            self._process_uploaded_file(uploaded_file)
    
    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file and display prediction interface."""
        try:
            df = pd.read_csv(uploaded_file)
            result_df = df.copy()
            
            self._display_data_preview(result_df)
            st.markdown("---")
            
            use_scaling = self._render_prediction_config()
            
            if st.button("🚀 Predict Sales", use_container_width=True, type="primary"):
                self._make_predictions(df, result_df, use_scaling)
            
            if 'last_prediction_csv' in st.session_state and st.session_state.last_prediction_csv:
                self._render_download_section()
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.exception(e)
    
    def _display_data_preview(self, df):
        """Display data preview and validation."""
        st.markdown("### 📋 Data Preview & Validation")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Uploaded Data Preview**")
            st.dataframe(df.head(100), use_container_width=True, height=400)
        
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
    
    def _render_prediction_config(self):
        """Render prediction configuration options."""
        st.markdown("### ⚙️ Prediction Configuration")
        col_a, col_b, col_c = st.columns([2, 1, 1])
        
        with col_a:
            st.markdown("**Configure your prediction settings**")
            use_scaling = st.checkbox(
                "Apply Feature Scaling",
                value=len(st.session_state.scalers) > 0,
                disabled=len(st.session_state.scalers) == 0,
                help="Apply loaded scalers to normalize features."
            )
            if len(st.session_state.scalers) == 0:
                st.caption("ℹ️ No scalers loaded. Scaling disabled.")
        
        with col_b:
            st.markdown("**Ready to predict?**")
        
        return use_scaling
    
    def _make_predictions(self, df, result_df, use_scaling):
        """Execute predictions and update session state."""
        model_features = st.session_state.model.input_shape[-1]
        
        if len(df.columns) != model_features:
            st.error(f"❌ Column mismatch! Expected {model_features} columns, got {len(df.columns)}")
            return
        
        with st.spinner("🔮 Predicting sales..."):
            try:
                predictions = self.predict_sales(df, use_scaling)
                result_df['predicted_sales'] = predictions
                
                st.session_state.last_prediction_csv = result_df.to_csv(index=False)
                
                st.markdown("### 📊 Data Preview (first 100 rows)")
                st.dataframe(result_df.head(100), use_container_width=True, height=400)
                
                st.session_state.prediction_history.append({
                    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rows': len(result_df),
                    'mean_sales': predictions.mean(),
                    'total_sales': predictions.sum()
                })
                
                st.success("✅ Predictions completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
    
    def _render_download_section(self):
        """Render download section."""
        st.markdown("---")
        st.markdown("#### 💾 Download Predictions")
        st.download_button(
            label="📥 Download CSV",
            data=st.session_state.last_prediction_csv,
            file_name=f"sales_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # ------------------------------------------------------------------------
    # UI RENDERING - HISTORY
    # ------------------------------------------------------------------------
    
    def render_history(self):
        """Render prediction history."""
        st.header("📜 Prediction History")
        st.markdown("### View your past predictions and session statistics")
        
        if st.session_state.prediction_history:
            self._render_history_content()
        else:
            self._render_empty_history()
    
    def _render_history_content(self):
        """Render history content with statistics."""
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        st.markdown("#### 📊 Session Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Total Rows Predicted", f"{history_df['rows'].sum():,}")
        with col3:
            st.metric("Avg Sales per Session", f"{history_df['mean_sales'].mean():,.2f}")
        
        st.markdown("---")
        st.markdown("#### 📋 Detailed History")
        st.dataframe(history_df, use_container_width=True, height=300)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                st.session_state.prediction_history = []
                st.session_state.last_prediction_df = None
                st.success("✅ History cleared!")
                st.rerun()
    
    def _render_empty_history(self):
        """Render message when history is empty."""
        st.markdown("---")
        st.info("""
        📭 **No prediction history yet!**
        
        To see your prediction history:
        1. Go to the "🎯 Predict Sales" tab
        2. Upload a CSV file
        3. Make predictions
        4. Your predictions will appear here
        """)
    
    # ------------------------------------------------------------------------
    # MAIN APPLICATION RUNNER
    # ------------------------------------------------------------------------
    
    def run(self):
        """Main application runner."""
        self._render_header()
        self.render_sidebar()
        
        tab1, tab2 = st.tabs(["🎯 Predict Sales", "📜 Prediction History"])
        
        with tab1:
            self.render_main_content()
        
        with tab2:
            self.render_history()
        
        self._render_footer()
    
    def _render_header(self):
        """Render application header."""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin-bottom: 0.5rem;'>📈 Sales Forecasting Dashboard</h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem;'>AI-Powered Sales Prediction Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_footer(self):
        """Render application footer."""
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


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = SalesPredictionApp()
    app.run()