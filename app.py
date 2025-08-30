"""
Fake News Detection Streamlit Application
Member 5: App Developer
File: app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .real-news {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .fake-news {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .confidence-high {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# App title and description
st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)

st.markdown("""
**Welcome to our AI-powered Fake News Detection System!**  
This application uses advanced Machine Learning and Natural Language Processing to analyze news statements and determine their credibility.

**Team Members:**
- Member 1: Data Cleaning (ITBIN-2211-0148)
- Member 2: TF-IDF Model (ITBIN-2211-0184)  
- Member 3: BERT Model (ITBIN-2211-0149)
- Member 4: Hybrid Model (ITBIN-2211-0173)
- Member 5: Streamlit App (ITBIN-2211-0169)
""")

# Sidebar for model information and settings
with st.sidebar:
    st.header("📊 Model Information")
    
    # Load model configuration
    try:
        with open('models/hybrid_config.json', 'r') as f:
            model_config = json.load(f)
        
        st.success("✅ Hybrid model loaded successfully!")
        st.write(f"**Model Type:** {model_config['combination_name']}")
        st.write(f"**Algorithm:** {model_config['model_name'].replace('_', ' ').title()}")
        st.write(f"**Test Accuracy:** {model_config['test_accuracy']:.3f}")
        
        if model_config.get('requires_bert', False):
            st.info("🤖 This model uses BERT embeddings")
        
    except FileNotFoundError:
        st.error("❌ Model configuration not found!")
        st.error("Please run Member 4's notebook first.")
        st.stop()
    
    st.markdown("---")
    st.header("🎛️ Settings")
    
    # Model selection option
    model_choice = st.selectbox(
        "Choose Model:",
        ["Hybrid Model (Best)", "TF-IDF Only", "Compare All"]
    )
    
    # Advanced options
    show_advanced = st.checkbox("Show Advanced Analysis")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)

# Load required models and utilities
@st.cache_resource
def load_models():
    """Load all required models and vectorizers"""
    models = {}
    
    try:
        # Load hybrid model components
        models['hybrid_model'] = joblib.load('models/hybrid_model.pkl')
        models['hybrid_scaler'] = joblib.load('models/hybrid_scaler.pkl')
        models['tfidf_vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Try to load TF-IDF only model
        try:
            models['tfidf_model'] = joblib.load('models/tfidf_model.pkl')
        except:
            st.warning("⚠️ TF-IDF only model not found")
        
        # Check if BERT is available
        if os.path.exists('models/saved_bert_model'):
            try:
                sys.path.append('utils')
                from bert_utils import load_bert_model
                bert_model, bert_tokenizer, device = load_bert_model('models/saved_bert_model')
                models['bert_model'] = bert_model
                models['bert_tokenizer'] = bert_tokenizer
                models['bert_device'] = device
            except Exception as e:
                st.warning(f"⚠️ BERT model loading failed: {e}")
        
        return models
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# Text cleaning function
def clean_text(text):
    """Clean input text (same as data cleaning)"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction functions
def predict_with_tfidf_only(statement, models):
    """Make prediction using TF-IDF model only"""
    try:
        cleaned = clean_text(statement)
        features = models['tfidf_vectorizer'].transform([cleaned])
        
        if 'tfidf_model' in models:
            prediction = models['tfidf_model'].predict(features)[0]
            probabilities = models['tfidf_model'].predict_proba(features)[0]
        else:
            # Use hybrid model with TF-IDF features only
            tfidf_features = features.toarray()
            # Pad with zeros for metadata if needed
            if hasattr(models['hybrid_model'], 'n_features_in_'):
                expected_features = models['hybrid_model'].n_features_in_
                if tfidf_features.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - tfidf_features.shape[1]))
                    tfidf_features = np.hstack([tfidf_features, padding])
            
            scaled_features = models['hybrid_scaler'].transform(tfidf_features)
            prediction = models['hybrid_model'].predict(scaled_features)[0]
            probabilities = models['hybrid_model'].predict_proba(scaled_features)[0]
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probabilities),
            'probabilities': {'Fake': probabilities[0], 'Real': probabilities[1]}
        }
    except Exception as e:
        return {'error': str(e)}

def predict_with_hybrid(statement, party, subject, models):
    """Make prediction using hybrid model with all features"""
    try:
        # Clean text
        cleaned = clean_text(statement)
        
        # Get TF-IDF features
        tfidf_features = models['tfidf_vectorizer'].transform([cleaned]).toarray()
        
        # Create metadata features
        party_map = {'republican': 0, 'democrat': 1, 'independent': 2, 'none': 3}
        subject_map = {'politics': 0, 'healthcare': 1, 'education': 2, 'economy': 3, 'other': 4}
        
        party_code = party_map.get(party.lower(), 3)
        subject_code = subject_map.get(subject.lower(), 4)
        
        # Calculate text statistics
        statement_length = len(cleaned)
        word_count = len(cleaned.split())
        has_question = 1 if '?' in statement else 0
        has_exclamation = 1 if '!' in statement else 0
        has_quotes = 1 if '"' in statement else 0
        uppercase_ratio = sum(c.isupper() for c in statement) / max(len(statement), 1)
        avg_word_length = np.mean([len(word) for word in cleaned.split()]) if cleaned.split() else 0
        
        # Combine metadata features
        meta_features = np.array([[
            party_code, subject_code, 0, 0, 0,  # Basic categorical codes
            statement_length, word_count, has_question, has_exclamation,
            has_quotes, uppercase_ratio, avg_word_length
        ]])
        
        # Combine all features
        combined_features = np.hstack([tfidf_features, meta_features])
        
        # Scale features
        scaled_features = models['hybrid_scaler'].transform(combined_features)
        
        # Make prediction
        prediction = models['hybrid_model'].predict(scaled_features)[0]
        probabilities = models['hybrid_model'].predict_proba(scaled_features)[0]
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probabilities),
            'probabilities': {'Fake': probabilities[0], 'Real': probabilities[1]},
            'features_used': {
                'text_features': tfidf_features.shape[1],
                'metadata_features': meta_features.shape[1],
                'total_features': combined_features.shape[1]
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

def predict_with_bert(statement, models):
    """Make prediction using BERT model"""
    try:
        if 'bert_model' not in models:
            return {'error': 'BERT model not available'}
        
        # Tokenize input
        inputs = models['bert_tokenizer'](
            statement,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Move to device
        inputs = {key: val.to(models['bert_device']) for key, val in inputs.items()}
        
        # Make prediction
        models['bert_model'].eval()
        with torch.no_grad():
            outputs = models['bert_model'](**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions.max().item()
        
        return {
            'prediction': 'Real' if predicted_class == 1 else 'Fake',
            'confidence': confidence,
            'probabilities': {
                'Fake': predictions[0][0].item(),
                'Real': predictions[0][1].item()
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

# Load models
models = load_models()
if models is None:
    st.stop()

# Main application interface
st.header("📰 News Statement Analysis")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Enter News Statement to Analyze:",
        height=150,
        placeholder="Paste your news headline or statement here...\n\nExample: 'The President announced new economic policies to boost employment rates.'"
    )

with col2:
    st.subheader("📋 Context Information")
    
    party_option = st.selectbox(
        "Speaker's Political Affiliation:",
        ['None', 'Republican', 'Democrat', 'Independent'],
        help="Select the political party of the statement's speaker"
    )
    
    subject_option = st.selectbox(
        "News Subject Category:",
        ['Other', 'Politics', 'Healthcare', 'Education', 'Economy'],
        help="Select the main topic category"
    )

# Analysis button
if st.button('🔍 Analyze News Statement', type="primary"):
    if user_input.strip():
        
        with st.spinner('🤖 Analyzing statement...'):
            
            # Make predictions based on selected model
            if model_choice == "Hybrid Model (Best)":
                result = predict_with_hybrid(user_input, party_option, subject_option, models)
            elif model_choice == "TF-IDF Only":
                result = predict_with_tfidf_only(user_input, models)
            else:  # Compare All
                results = {}
                results['hybrid'] = predict_with_hybrid(user_input, party_option, subject_option, models)
                results['tfidf'] = predict_with_tfidf_only(user_input, models)
                if 'bert_model' in models:
                    results['bert'] = predict_with_bert(user_input, models)
        
        # Display results
        if model_choice != "Compare All":
            # Single model result
            if 'error' not in result:
                # Main prediction display
                prediction = result['prediction']
                confidence = result['confidence']
                
                if prediction == 'Real':
                    st.markdown(f"""
                    <div class="prediction-box real-news">
                        ✅ This statement appears to be <strong>REAL NEWS</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box fake-news">
                        ❌ This statement appears to be <strong>FAKE NEWS</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence display
                if show_confidence:
                    conf_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
                    st.markdown(f"""
                    <div class="prediction-box {conf_class}">
                        📊 Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔴 Fake Probability", f"{result['probabilities']['Fake']:.1%}")
                    with col2:
                        st.metric("🟢 Real Probability", f"{result['probabilities']['Real']:.1%}")
                    with col3:
                        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                        st.metric("📈 Confidence Level", confidence_level)
                
                # Advanced analysis
                if show_advanced and 'features_used' in result:
                    st.subheader("🔧 Technical Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Features Used:**")
                        st.write(f"- Text Features: {result['features_used']['text_features']}")
                        st.write(f"- Metadata Features: {result['features_used']['metadata_features']}")
                        st.write(f"- Total Features: {result['features_used']['total_features']}")
                    
                    with col2:
                        st.write("**Input Analysis:**")
                        st.write(f"- Character Count: {len(user_input)}")
                        st.write(f"- Word Count: {len(user_input.split())}")
                        st.write(f"- Party: {party_option}")
                        st.write(f"- Subject: {subject_option}")
                
                # Store in history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'statement': user_input[:100] + '...' if len(user_input) > 100 else user_input,
                    'prediction': prediction,
                    'confidence': confidence,
                    'party': party_option,
                    'subject': subject_option
                })
                
            else:
                st.error(f"❌ Prediction error: {result['error']}")
        
        else:
            # Compare all models
            st.subheader("📊 Model Comparison Results")
            
            comparison_data = []
            valid_results = {}
            
            for model_name, model_result in results.items():
                if 'error' not in model_result:
                    comparison_data.append({
                        'Model': model_name.title(),
                        'Prediction': model_result['prediction'],
                        'Confidence': f"{model_result['confidence']:.1%}",
                        'Real Prob': f"{model_result['probabilities']['Real']:.1%}",
                        'Fake Prob': f"{model_result['probabilities']['Fake']:.1%}"
                    })
                    valid_results[model_name] = model_result
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization of model agreement
                predictions = [r['prediction'] for r in valid_results.values()]
                confidences = [r['confidence'] for r in valid_results.values()]
                model_names = list(valid_results.keys())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model predictions chart
                    pred_counts = pd.Series(predictions).value_counts()
                    fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                               title="Model Predictions Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confidence comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=model_names,
                        y=confidences,
                        text=[f"{c:.1%}" for c in confidences],
                        textposition='auto'
                    ))
                    fig.update_layout(title="Model Confidence Comparison", 
                                    yaxis_title="Confidence Score")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("❌ No valid model predictions obtained")
    else:
        st.warning("⚠️ Please enter a news statement to analyze")

# Sample statements for testing
st.header("🧪 Try Sample Statements")

sample_statements = {
    "Real News Example": "The Federal Reserve announced a 0.25% interest rate increase to combat inflation.",
    "Suspicious Example": "Scientists have discovered that drinking soda every day actually makes you live longer!",
    "Political Example": "Congressional leaders reached a bipartisan agreement on infrastructure spending.",
    "Health Example": "New clinical trial shows promising results for Alzheimer's treatment."
}

cols = st.columns(len(sample_statements))
for i, (label, statement) in enumerate(sample_statements.items()):
    with cols[i]:
        if st.button(f"📰 {label}", key=f"sample_{i}"):
            # Auto-fill the text area
            st.rerun()

# Prediction history
if st.session_state.prediction_history:
    st.header("📈 Prediction History")
    
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    if not history_df.empty:
        # Display recent predictions
        st.subheader("Recent Predictions")
        
        for idx, row in history_df.tail(5).iterrows():
            with st.expander(f"{row['prediction']} - {row['timestamp'].strftime('%H:%M:%S')} - Confidence: {row['confidence']:.1%}"):
                st.write(f"**Statement:** {row['statement']}")
                st.write(f"**Party:** {row['party']}")
                st.write(f"**Subject:** {row['subject']}")
                st.write(f"**Prediction:** {row['prediction']}")
                st.write(f"**Confidence:** {row['confidence']:.1%}")
        
        # History analytics
        if len(history_df) > 1:
            st.subheader("📊 Usage Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fake_count = sum(history_df['prediction'] == 'Fake')
                real_count = sum(history_df['prediction'] == 'Real')
                
                fig = px.pie(
                    values=[fake_count, real_count],
                    names=['Fake', 'Real'],
                    title="Prediction Distribution",
                    color_discrete_map={'Fake': 'red', 'Real': 'green'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_confidence = history_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
                
                high_conf = sum(history_df['confidence'] > 0.8)
                st.metric("High Confidence Predictions", f"{high_conf}/{len(history_df)}")
            
            with col3:
                subject_counts = history_df['subject'].value_counts()
                fig = px.bar(
                    x=subject_counts.index,
                    y=subject_counts.values,
                    title="Subjects Analyzed"
                )
                st.plotly_chart(fig, use_container_width=True)

# Model performance section
st.header("📊 Model Performance Dashboard")

try:
    # Load performance data
    performance_df = pd.read_csv('models/performance_benchmark.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Models")
        top_models = performance_df.head(10)[['Approach', 'Test_Accuracy', 'Complexity']]
        
        # Color code by complexity
        def color_complexity(val):
            if val == 'Low':
                return 'background-color: #d4edda'
            elif val == 'Medium':
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        styled_df = top_models.style.applymap(color_complexity, subset=['Complexity'])
        st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Performance Visualization")
        
        fig = px.scatter(
            performance_df,
            x='Feature_Count',
            y='Test_Accuracy',
            color='Complexity',
            hover_data=['Approach'],
            title="Accuracy vs Model Complexity",
            log_x=True
        )
        st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.info("📊 Performance data will be available after running all team notebooks")

# Information and help section
with st.expander("ℹ️ How to Use This Application"):
    st.markdown("""
    **Step 1:** Enter a news statement in the text area above
    
    **Step 2:** Provide context information:
    - Select the speaker's political affiliation (if known)
    - Choose the subject category that best fits the news
    
    **Step 3:** Click "Analyze News Statement" to get results
    
    **Understanding Results:**
    - ✅ **Real News**: The statement appears credible based on our analysis
    - ❌ **Fake News**: The statement shows characteristics of misinformation
    - 📊 **Confidence**: How certain the model is about its prediction
    
    **Model Features:**
    - 📝 **Text Analysis**: Advanced NLP processing of statement content
    - 📊 **Metadata Analysis**: Context like political affiliation and subject
    - 🤖 **Hybrid Approach**: Combines multiple AI techniques for better accuracy
    """)

with st.expander("🔬 About Our Models"):
    st.markdown("""
    **Our Team developed three complementary approaches:**
    
    **1. TF-IDF Model (Member 2)**
    - Traditional machine learning approach
    - Analyzes word frequency and importance
    - Fast and interpretable baseline model
    
    **2. BERT Model (Member 3)**  
    - State-of-the-art transformer neural network
    - Understands context and semantic meaning
    - Highest accuracy for text-only analysis
    
    **3. Hybrid Model (Member 4)**
    - Combines text analysis with metadata features
    - Uses both content and context for decisions
    - Best overall performance across different news types
    
    **Data Processing (Member 1)**
    - Cleaned and preprocessed LIAR dataset
    - 12,836 labeled statements from public figures
    - Rich metadata including political affiliations and subjects
    """)

# Footer with team information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <h4>🎓 Fake News Detection Project</h4>
    <p>Developed by Team Members: ITBIN-2211-0148, ITBIN-2211-0184, ITBIN-2211-0149, ITBIN-2211-0173, ITBIN-2211-0169</p>
    <p>Using Advanced NLP, Machine Learning, and Metadata Analysis</p>
    <p><em>Built with Streamlit, scikit-learn, BERT, and ❤️</em></p>
</div>
""", unsafe_allow_html=True)

# Debug information (only show if models are not working)
if st.checkbox("🔧 Show Debug Information", value=False):
    st.subheader("Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Loaded Models:**")
        for model_name in models.keys():
            st.write(f"✅ {model_name}")
    
    with col2:
        st.write("**File Status:**")
        required_files = [
            'models/hybrid_model.pkl',
            'models/hybrid_scaler.pkl',
            'models/tfidf_vectorizer.pkl',
            'models/hybrid_config.json'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                st.write(f"✅ {file_path}")
            else:
                st.write(f"❌ {file_path}")
    
    st.write("**Model Configuration:**")
    try:
        with open('models/hybrid_config.json', 'r') as f:
            config = json.load(f)
        st.json(config)
    except:
        st.error("Configuration file not found")

# Real-time model stats (if available)
try:
    if os.path.exists('models/hybrid_model_results.csv'):
        with st.expander("📈 Detailed Model Statistics"):
            stats_df = pd.read_csv('models/hybrid_model_results.csv')
            st.dataframe(stats_df, use_container_width=True)
            
            # Model performance chart
            if len(stats_df) > 1:
                fig = px.bar(
                    stats_df.head(10),
                    x='Test_Accuracy',
                    y='Approach',
                    orientation='h',
                    title="Model Performance Comparison",
                    color='Test_Accuracy',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
except Exception as e:
    pass  # Stats not available yet

# Batch analysis feature
st.header("📚 Batch Analysis")
st.write("Upload a CSV file with news statements for batch analysis")

uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=['csv'],
    help="CSV should have 'statement' column and optionally 'party' and 'subject' columns"
)

if uploaded_file is not None:
    try:
        # Read uploaded file
        batch_df = pd.read_csv(uploaded_file)
        
        st.write(f"📁 Loaded {len(batch_df)} statements for analysis")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("🚀 Analyze All Statements"):
            
            progress_bar = st.progress(0)
            results_list = []
            
            for idx, row in batch_df.iterrows():
                # Update progress
                progress_bar.progress((idx + 1) / len(batch_df))
                
                # Get statement and metadata
                statement = row.get('statement', '')
                party = row.get('party', 'none')
                subject = row.get('subject', 'other')
                
                # Make prediction
                result = predict_with_hybrid(statement, party, subject, models)
                
                if 'error' not in result:
                    results_list.append({
                        'Original_Statement': statement[:100] + '...' if len(statement) > 100 else statement,
                        'Prediction': result['prediction'],
                        'Confidence': result['confidence'],
                        'Real_Probability': result['probabilities']['Real'],
                        'Fake_Probability': result['probabilities']['Fake']
                    })
                else:
                    results_list.append({
                        'Original_Statement': statement,
                        'Prediction': 'Error',
                        'Confidence': 0,
                        'Real_Probability': 0,
                        'Fake_Probability': 0
                    })
            
            # Display results
            results_df = pd.DataFrame(results_list)
            
            st.success(f"✅ Analyzed {len(results_df)} statements!")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fake_count = sum(results_df['Prediction'] == 'Fake')
                st.metric("🔴 Fake News", fake_count)
            
            with col2:
                real_count = sum(results_df['Prediction'] == 'Real')
                st.metric("🟢 Real News", real_count)
            
            with col3:
                avg_conf = results_df['Confidence'].mean()
                st.metric("📊 Avg Confidence", f"{avg_conf:.1%}")
            
            with col4:
                high_conf = sum(results_df['Confidence'] > 0.8)
                st.metric("🎯 High Confidence", f"{high_conf}/{len(results_df)}")
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Results CSV",
                data=csv,
                file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV has a 'statement' column")

# Educational section
st.header("🎓 Understanding Fake News Detection")

tab1, tab2, tab3 = st.tabs(["🧠 How It Works", "📊 Model Features", "⚠️ Limitations"])

with tab1:
    st.markdown("""
    **Our AI System Uses Multiple Techniques:**
    
    **1. Natural Language Processing (NLP)**
    - Analyzes the actual words and phrases in the statement
    - Looks for patterns commonly found in fake vs real news
    - Uses both traditional (TF-IDF) and modern (BERT) approaches
    
    **2. Metadata Analysis**
    - Considers who made the statement (political party)
    - Analyzes the subject matter (politics, health, etc.)
    - Uses historical patterns from similar sources
    
    **3. Hybrid Intelligence**
    - Combines text analysis with contextual information
    - Makes more informed decisions than text-only systems
    - Learns from 12,836+ verified examples
    """)

with tab2:
    st.markdown("""
    **Text Features Analyzed:**
    - Word frequency and importance (TF-IDF)
    - Semantic meaning and context (BERT embeddings)
    - Statement length and complexity
    - Punctuation patterns (questions, exclamations)
    - Writing style indicators
    
    **Metadata Features:**
    - Speaker's political affiliation
    - News subject category
    - Historical credibility patterns
    - Source reliability indicators
    
    **Advanced Features:**
    - Cross-feature interactions
    - Ensemble model predictions
    - Confidence calibration
    """)

with tab3:
    st.markdown("""
    **Important Limitations:**
    
    **⚠️ This is a Research Tool**
    - Not a substitute for critical thinking
    - Should not be the only factor in evaluating news
    - May have biases from training data
    
    **🔍 Always Verify:**
    - Check multiple reliable sources
    - Look for original sources and citations
    - Consider the publication's track record
    - Be aware of your own confirmation bias
    
    **🎯 Best Used For:**
    - Initial screening of suspicious content
    - Educational purposes and research
    - Understanding patterns in misinformation
    - Supporting fact-checking workflows
    """)

# Performance monitoring (for team/instructor)
if st.checkbox("👨‍🏫 Show Performance Monitoring", value=False):
    st.subheader("🔧 System Performance")
    
    # Model loading status
    model_status = {}
    model_files = [
        ('Hybrid Model', 'models/hybrid_model.pkl'),
        ('TF-IDF Vectorizer', 'models/tfidf_vectorizer.pkl'),
        ('Hybrid Scaler', 'models/hybrid_scaler.pkl'),
        ('BERT Model', 'models/saved_bert_model/config.json')
    ]
    
    for name, path in model_files:
        model_status[name] = "✅ Loaded" if os.path.exists(path) else "❗ Missing"
    
    status_df = pd.DataFrame(list(model_status.items()), columns=['Component', 'Status'])
    st.dataframe(status_df, use_container_width=True)
    
    # System info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**System Information:**")
        st.write(f"- Python: {sys.version.split()[0]}")
        st.write(f"- Streamlit: {st.__version__}")
        st.write(f"- Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.write("**Session Statistics:**")
        st.write(f"- Predictions Made: {len(st.session_state.prediction_history)}")
        if st.session_state.prediction_history:
            recent_accuracy = np.mean([h['confidence'] for h in st.session_state.prediction_history[-10:]])
            st.write(f"- Recent Avg Confidence: {recent_accuracy:.1%}")