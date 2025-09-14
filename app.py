import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="🏏 IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header { 
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏏 IPL Score Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict the final score of an IPL match based on current match situation")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    model = model_data['model']
    feature_columns = model_data.get('feature_columns', [])
    
    # Sidebar for match information
    with st.sidebar:
        st.header("🏏 Match Configuration")
        
        # Team selection
        st.subheader("Teams")
        teams = [
            'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
            'Royal Challengers Bangalore', 'Delhi Capitals', 'Punjab Kings',
            'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans',
            'Lucknow Super Giants'
        ]
        
        batting_team = st.selectbox("🏏 Batting Team", teams, key="batting")
        bowling_team = st.selectbox("⚡ Bowling Team", 
                                   [team for team in teams if team != batting_team], 
                                   key="bowling")
        
        # Venue selection
        st.subheader("Venue")
        venues = [
            'M Chinnaswamy Stadium, Bangalore',
            'Wankhede Stadium, Mumbai', 
            'Eden Gardens, Kolkata',
            'Feroz Shah Kotla, Delhi',
            'MA Chidambaram Stadium, Chennai',
            'Rajiv Gandhi Intl Stadium, Hyderabad',
            'Sawai Mansingh Stadium, Jaipur',
            'Narendra Modi Stadium, Ahmedabad',
            'Ekana Cricket Stadium, Lucknow',
            'Punjab Cricket Association Stadium, Mohali'
        ]
        
        venue = st.selectbox("🏟️ Venue", venues)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Match Situation")
        
        # Match situation inputs
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            overs = st.number_input(
                "⏱️ Overs Completed", 
                min_value=5.0, max_value=19.5, 
                value=10.0, step=0.1,
                help="Number of overs completed (minimum 5.0)"
            )
            
            current_runs = st.number_input(
                "🏃 Current Runs", 
                min_value=0, max_value=300, 
                value=80, step=1,
                help="Total runs scored so far"
            )
            
            wickets = st.number_input(
                "❌ Wickets Lost", 
                min_value=0, max_value=10, 
                value=2, step=1,
                help="Number of wickets fallen"
            )
        
        with col1_2:
            runs_last_5 = st.number_input(
                "🔥 Runs in Last 5 Overs", 
                min_value=0, max_value=150, 
                value=35, step=1,
                help="Runs scored in the last 5 overs"
            )
            
            wickets_last_5 = st.number_input(
                "💥 Wickets in Last 5 Overs", 
                min_value=0, max_value=5, 
                value=1, step=1,
                help="Wickets lost in the last 5 overs"
            )
    
    with col2:
        st.subheader("📈 Match Metrics")
        
        # Calculate current metrics
        current_rr = current_runs / overs if overs > 0 else 0
        balls_left = (20 - overs) * 6
        
        st.markdown(f"""
        <div class="metric-container">
            <strong>Current Run Rate:</strong><br>
            <span style="font-size: 1.5em; color: #1f77b4;">{current_rr:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <strong>Balls Remaining:</strong><br>
            <span style="font-size: 1.5em; color: #ff7f0e;">{int(balls_left)}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <strong>Run Rate (Last 5):</strong><br>
            <span style="font-size: 1.5em; color: #2ca02c;">{runs_last_5/5:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("---")
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        predict_button = st.button(
            "🔮 PREDICT FINAL SCORE", 
            type="primary",
            use_container_width=True
        )
    
    # Make prediction
    if predict_button:
        try:
            # Prepare input data
            input_data = prepare_prediction_data(
                batting_team, bowling_team, venue, overs, 
                current_runs, wickets, runs_last_5, wickets_last_5,
                feature_columns
            )
            
            # Make prediction
            predicted_score = model.predict(input_data)[0]
            predicted_score = max(current_runs, int(predicted_score))  # Ensure >= current runs
            
            # Display prediction
            st.markdown("---")
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #1f77b4;">
                    🎯 PREDICTED FINAL SCORE: {predicted_score} RUNS
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            runs_needed = predicted_score - current_runs
            required_rr = (runs_needed / balls_left) * 6 if balls_left > 0 else 0
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("🏃 Runs Needed", f"{runs_needed}")
            
            with metric_col2:
                st.metric("📈 Required Run Rate", f"{required_rr:.2f}")
            
            with metric_col3:
                st.metric("⚡ Current Run Rate", f"{current_rr:.2f}")
            
            with metric_col4:
                st.metric("🎯 Predicted Range", f"{predicted_score-15} - {predicted_score+15}")
            
            # Prediction confidence
            if required_rr <= 6:
                confidence = "High"
                conf_color = "green"
            elif required_rr <= 8:
                confidence = "Medium"
                conf_color = "orange"
            else:
                confidence = "Low"
                conf_color = "red"
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <span style="color: {conf_color}; font-weight: bold;">
                    Prediction Confidence: {confidence}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🤖 About this Model")
    
    with st.expander("Model Information"):
        st.markdown("""
        **Algorithms Used:**
        - 🌳 Random Forest Regressor
        - 🚀 XGBoost Regressor  
        - 📊 Linear Regression
        - 🎯 K-Nearest Neighbors
        - ⚡ Support Vector Regression
        - 🌿 Decision Tree Regressor
        
        **Features:**
        - Historical IPL data (2008-2020+)
        - Team performance analysis
        - Venue-specific predictions
        - Real-time match situation assessment
        
        **Optimization:**
        - Hyperparameter tuning with Optuna
        - Cross-validation for robust performance
        - Feature engineering for better accuracy
        """)

def prepare_prediction_data(batting_team, bowling_team, venue, overs, runs, wickets, 
                          runs_last_5, wickets_last_5, feature_columns):
    """Prepare input data for prediction"""
    
    # Create base data
    data = {
        'runs': runs,
        'wickets': wickets, 
        'overs': overs,
        'runs_last_5': runs_last_5,
        'wickets_last_5': wickets_last_5
    }
    
    # Add one-hot encoded team columns
    teams = ['Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
             'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
             'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
    
    # Initialize all team columns to 0
    for team in teams:
        data[f'batting_team_{team}'] = 0
        data[f'bowling_team_{team}'] = 0
    
    # Set the selected teams to 1
    if f'batting_team_{batting_team}' in data:
        data[f'batting_team_{batting_team}'] = 1
    if f'bowling_team_{bowling_team}' in data:
        data[f'bowling_team_{bowling_team}'] = 1
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Ensure we have all required columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_columns]

if __name__ == "__main__":
    main()
 