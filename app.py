import streamlit as st
import joblib
import numpy as np
import streamlit.components.v1 as components
import os
import pandas as pd

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Main background is handled by Streamlit theme */
.header-container {
    background: #262730;
    color: #FAFAFA;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    background-color: #222;
    color: #FAFAFA;
}
.real-box { border-left: 8px solid #28a745; }
.fake-box { border-left: 8px solid #dc3545; }
.info-card {
    background-color: #222;
    color: #FAFAFA;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.15);
}
.stTextArea>div>div>textarea {
    background-color: #262730;
    color: #FAFAFA;
    border: 1px solid #444;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
    padding: 0.5rem 2rem;
}
</style>
""", unsafe_allow_html=True)


# Function to load models safely
@st.cache_resource
def load_models():
    try:
        model = joblib.load('ensemble_fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

# Function to make predictions
def predict_news(news_text, model, vectorizer):
    try:
        # Transform the text
        news_vector = vectorizer.transform([news_text])
        
        # Get prediction and probability
        prediction = model.predict(news_vector)[0]
        probabilities = model.predict_proba(news_vector)[0]
        
        # Get confidence
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        return {
            "prediction": "REAL" if prediction == 1 else "FAKE",
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# App Header
st.markdown("""
<div class="header-container">
    <div style="margin-right: 20px; font-size: 3rem;">📰</div>
    <div>
        <h1>Fake News Detector</h1>
        <p>Powered by Machine Learning</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
model, vectorizer = load_models()

if model is not None and vectorizer is not None:
    # Main layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # News text input
        news_text = st.text_area(
            "Paste news headline or article text below:",
            height=200,
            placeholder="Enter text here to analyze..."
        )
        
        # Analysis button
        analyze_button = st.button("Check Authenticity")
        
        # Session state for history
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Make prediction when button is clicked
        if analyze_button:
            if news_text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing content..."):
                    result = predict_news(news_text, model, vectorizer)
                    
                    if result:
                        # Add to history
                        st.session_state.history.append({
                            "text": news_text,
                            "result": result
                        })
                        
                        # Display result with fancy styling
                        box_class = "real-box" if result["prediction"] == "REAL" else "fake-box"
                        icon = "✅" if result["prediction"] == "REAL" else "⚠️"
                        confidence_percentage = f"{result['confidence'] * 100:.1f}%"
                        
                        st.markdown(f"""
                        <div class="result-box {box_class}">
                            <h2>{icon} This news appears to be {result["prediction"]}</h2>
                            <p>Confidence: {confidence_percentage}</p>
                            <div class="confidence-meter">
                                <div style="width:{confidence_percentage}; height:100%; 
                                    background-color:{'#28a745' if result['prediction'] == 'REAL' else '#dc3545'}; 
                                    border-radius:5px;">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probabilities chart
                        probabilities = result["probabilities"]
                        chart_data = pd.DataFrame({
                            "Category": ["Fake", "Real"],
                            "Probability": [probabilities[0], probabilities[1]]
                        })
                        st.subheader("Prediction Probabilities")
                        st.bar_chart(chart_data.set_index("Category"))
    
    with col2:
        # Info card
        st.markdown("""
        <div class="info-card">
            <h3>How It Works</h3>
            <p>Our AI model analyzes news content for patterns that indicate potential misinformation.</p>
            <p>The model was trained on thousands of verified real and fake news articles.</p>
            <p>Current model accuracy: <strong>92%</strong></p>
            <hr>
            <h4>Tips for Best Results:</h4>
            <ul>
                <li>Include full headlines</li>
                <li>Longer articles provide better accuracy</li>
                <li>Copy text directly from the source</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # History Section
        if st.session_state.history:
            st.markdown("<h3>Recent Checks</h3>", unsafe_allow_html=True)
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                with st.expander(f"{item['result']['prediction']}: {item['text'][:50]}..."):
                    st.write(item['text'])
                    st.write(f"Prediction: {item['result']['prediction']} ({item['result']['confidence'] * 100:.1f}%)")
else:
    # Error message if models couldn't be loaded
    st.error("""
    Model files not found or could not be loaded. Please make sure:
    
    1. Both 'ensemble_fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this app
    2. You have proper permissions to read these files
    3. The files aren't corrupted
    
    If you're running this app on Streamlit Cloud, make sure to upload these files to your GitHub repository.
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; font-size: 0.8rem; color: #6c757d;">
    <p>Fake News Detector &copy; 2025 | Model accuracy: 92% | Not to be used as the sole source for determining news authenticity</p>
</div>
""", unsafe_allow_html=True)
