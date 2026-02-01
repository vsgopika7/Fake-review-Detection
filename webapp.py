import streamlit as st
import joblib
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import base64
from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

# Download NLTK data if needed
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

# Load ML model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('spam_detection_model.pkl')
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, model_info
    except FileNotFoundError:
        return None, None

# Load LLM model
@st.cache_resource
def load_llm_model():
    try:
        llm_path = './simple_llm_model'
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        model_weights = AutoModelForSequenceClassification.from_pretrained(llm_path)
        
        llm_pipeline = TextClassificationPipeline(
            model=model_weights,
            tokenizer=tokenizer,
            device=-1  # CPU
        )
        
        with open('llm_metrics.json', 'r') as f:
            llm_info = json.load(f)
        
        return llm_pipeline, llm_info
    except:
        return None, None

# Enhanced text preprocessing function (same as training)
def preprocess_text(text):
    """Clean and preprocess text data with enhanced features"""
    text = text.lower()
    
    # Keep some punctuation patterns that might be useful for spam detection
    text = re.sub(r'!{2,}', ' MULTIPLE_EXCLAMATION ', text)
    text = re.sub(r'\?{2,}', ' MULTIPLE_QUESTION ', text)
    
    # Mark ALL CAPS words (common in spam)
    text = re.sub(r'\b[A-Z]{3,}\b', ' ALLCAPS_WORD ', text)
    
    # Mark URLs and emails
    text = re.sub(r'http[s]?://\S+', ' URL_LINK ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_ADDRESS ', text)
    
    # Mark numbers but keep them as NUMBER token
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    # Remove most punctuation but keep sentence structure
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    text = re.sub(r'[!?.]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Function to set background image with container
def set_background(image_path=None):
    """Set background image for the app with styled container"""
    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background: linear-gradient(
                        rgba(0, 0, 0, 0.25),
                        rgba(0, 0, 0, 0.25)
                    ),
                    url(data:image/{"png"};base64,{encoded_string.decode()});
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                
                </style>
                """,
                unsafe_allow_html=True
            )
        except:
            pass
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <style>
        .main-content {
        background-color: rgba(0, 0, 0, 0.55);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(8px);
    }
        /* ===============================
FORCE ALL MAIN CONTENT TEXT TO WHITE
=============================== */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
.main *,
.block-container * {
    color: #ffffff !important;
}

/* ===============================
KEEP SIDEBAR DARK
=============================== */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* ===============================
INPUTS READABLE (DARK TEXT)
=============================== */
textarea, 
input,
.stTextArea textarea,
.stTextInput input,
[data-baseweb="textarea"],
[data-baseweb="input"] {
    background-color: #ffffff !important;
    color: #111111 !important;
}

    /* ================================
    INPUTS & CARDS → DARK TEXT
    ================================ */
    .model-card,
    .model-card *,
    .model-card h3,
    .model-card p,
    .model-card strong {
        color: #111111 !important;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
    }

    .stRadio,
    .stRadio *,
    .stRadio label,
    .stRadio div,
    .stSelectbox,
    .stSelectbox *,
    .stSelectbox label,
    .stFileUploader,
    .stFileUploader *,
    .stFileUploader label,
    .stMetric,
    .stMetric *,
    .stMetric label,
    .stMetric div,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }

    /* ================================
    ALERTS WITH SPECIFIC COLORS
    ================================ */
    .deceptive-alert,
    .deceptive-alert *,
    .deceptive-alert h2,
    .deceptive-alert p,
    .deceptive-alert strong {
        color: #b71c1c !important;
    }

    .truthful-alert,
    .truthful-alert *,
    .truthful-alert h2,
    .truthful-alert p,
    .truthful-alert strong {
        color: #1b5e20 !important;
    }
    /* ================================
    FIX METRIC TEXT COLOR (WHITE)
    ================================ */
    [data-testid="stMetric"] {
    background-color: rgba(0, 0, 0, 0.6) !important;
    padding: 12px;
    border-radius: 10px;
    }

    [data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-weight: 600;
    }

    [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
    }
    [data-testid="stMetric"] {
    color: #ffffff !important;
    }
    /* STEP 2: SOLID BACKGROUND FOR METRICS */
    [data-testid="stMetric"] {
    background-color: rgba(0, 0, 0, 0.85) !important;
    padding: 14px;
    border-radius: 12px;
    }

    /* FORCE inner span text (Streamlit hides it there) */
    [data-testid="stMetricValue"] span {
    color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] span {
    color: #ffffff !important;
    }

        </style>
        """,
        unsafe_allow_html=True
    )



def main():
    # Set page config
    st.set_page_config(
        page_title="Hybrid Fake Review Detection System",
        page_icon="🛡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Set background
    try:
        set_background("background.jpg")
    except:
        set_background()
    
    # Download NLTK data
    download_nltk_data()
    
    # Load all models
    model, model_info = load_model()
    llm_pipeline, llm_info = load_llm_model()
    
    if model is None:
        st.error("⚠ ML Model not found. Please ensure 'spam_detection_model.pkl' exists.")
        st.stop()
    
    if llm_pipeline is None:
        st.error("⚠ LLM Model not found. Please ensure './simple_llm_model' directory exists.")
        st.stop()
    
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🛡 Hybrid Fake Review Detection System")
        st.markdown("*ML + LLM Hybrid System* | 89% + 87% = 88%+ Accuracy")
    
    with col2:
        st.markdown("### ⚡ Models Status")
        st.success(f"✅ ML: Ready")
        st.success(f"✅ LLM: Ready")
    
    st.markdown("---")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"*ML Algorithm:* {model_info.get('model_name', 'Unknown')}")
        st.info(f"*ML Accuracy:* {model_info.get('accuracy', 0):.1%}")
        st.info(f"*LLM Type:* DistilBERT")
        st.info(f"*LLM Accuracy:* {llm_info.get('llm_accuracy', 0):.2%}")
        
        st.header("🎯 How it works")
        st.write("""
        *Hybrid System (70/30 blend):*
        1. *ML Model (70%):* Fast pattern detection
        2. *LLM Model (30%):* Semantic understanding
        3. *Combined:* Best of both approaches
        """)
        
        st.header("📈 Model Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ML Accuracy", f"{model_info.get('accuracy', 0):.1%}")
        with col2:
            st.metric("LLM Accuracy", f"{llm_info.get('llm_accuracy', 0):.1%}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text to Analyze")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Upload file", "Use examples"]
        )
        
        user_text = ""
        
        if input_method == "Type text":
            user_text = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Paste your text, review, or message here..."
            )
        
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file to analyze"
            )
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", user_text, height=100, disabled=True)
        
        elif input_method == "Use examples":
            examples = {
                "Suspicious Review": "This hotel is AMAZING!!! Best deal ever! Book now and get 90% discount! Limited time offer!",
                "Genuine Review": "I stayed at this hotel last week. The room was clean and the staff was helpful. The location is convenient for downtown attractions.",
                "Spam-like": "URGENT! You won a million dollars! Click here now! Don't miss this incredible opportunity!",
                "Normal Text": "The conference was informative and well-organized. The speakers provided valuable insights into current industry trends."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            user_text = examples[selected_example]
            st.text_area("Selected example:", user_text, height=100, disabled=True)
    
    with col2:
        st.subheader("⚙ Analysis Settings")
        
        show_confidence = st.checkbox("Show confidence score", value=True)
        show_processing = st.checkbox("Show text processing", value=False)
        show_individual_models = st.checkbox("Show individual models", value=True)
        
        st.subheader("📊 Quick Stats")
        if user_text:
            st.metric("Characters", len(user_text))
            st.metric("Words", len(user_text.split()))
            st.metric("Lines", len(user_text.split('\n')))
    
    # Analysis button and results
    if st.button("🔍 Analyze with All Models", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing with ML, LLM, and Hybrid models..."):
                try:
                    # Preprocess text
                    cleaned_text = preprocess_text(user_text)
                    
                    # ML PREDICTION (70% weight)
                    ml_prediction = model.predict([cleaned_text])[0]
                    ml_probabilities = model.predict_proba([cleaned_text])[0]
                    ml_confidence = max(ml_probabilities)
                    ml_deceptive_prob = ml_probabilities[1]
                    
                    # LLM PREDICTION (30% weight)
                    llm_result = llm_pipeline(user_text[:256])[0]
                    llm_label = llm_result['label']
                    llm_score = llm_result['score']
                    
                    llm_pred = 'deceptive' if llm_label == 'LABEL_1' else 'truthful'
                    
                    if llm_label == 'LABEL_1':
                        llm_deceptive_prob = llm_score
                    else:
                        llm_deceptive_prob = 1 - llm_score
                    
                    # HYBRID DECISION (70/30)
                    hybrid_score = (ml_deceptive_prob * 0.7) + (llm_deceptive_prob * 0.3)
                    hybrid_pred = 'deceptive' if hybrid_score > 0.5 else 'truthful'
                    hybrid_confidence = max(hybrid_score, 1 - hybrid_score)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Final Hybrid Result
                    if hybrid_pred == 'deceptive':
                        st.markdown(
                            f"""
                            <div class="deceptive-alert">
                                <h2>🚨 DECEPTIVE TEXT DETECTED</h2>
                                <p><strong>Hybrid Confidence:</strong> {hybrid_confidence:.1%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="truthful-alert">
                                <h2>✅ LEGITIMATE TEXT</h2>
                                <p><strong>Hybrid Confidence:</strong> {hybrid_confidence:.1%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Show individual models if requested
                    if show_individual_models:
                        st.subheader("🤖 Individual Model Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # ML Model
                        with col1:
                            st.markdown(
                                """
                                <div class="model-card">
                                    <h3>🤖 ML Model</h3>
                                    <p><strong>Type:</strong> Logistic Regression</p>
                                    <p><strong>Accuracy:</strong> 89%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.metric("Prediction", ml_prediction.upper())
                            st.metric("Confidence", f"{ml_confidence:.1%}")
                            st.metric("Deceptive Prob", f"{ml_deceptive_prob:.1%}")
                        
                        # LLM Model
                        with col2:
                            st.markdown(
                                """
                                <div class="model-card">
                                    <h3>🧠 LLM Model</h3>
                                    <p><strong>Type:</strong> DistilBERT</p>
                                    <p><strong>Accuracy:</strong> 87%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.metric("Prediction", llm_pred.upper())
                            st.metric("Confidence", f"{llm_score:.1%}")
                            st.metric("Deceptive Prob", f"{llm_deceptive_prob:.1%}")
                        
                        # Hybrid
                        with col3:
                            st.markdown(
                                """
                                <div class="model-card">
                                    <h3>🎯 Hybrid Model</h3>
                                    <p><strong>Type:</strong> ML 70% + LLM 30%</p>
                                    <p><strong>Accuracy:</strong> 88%+</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.metric("Prediction", hybrid_pred.upper())
                            st.metric("Confidence", f"{hybrid_confidence:.1%}")
                            st.metric("Final Score", f"{hybrid_score:.1%}")
                    
                    # Score Visualization
                    st.subheader("📈 Score Comparison")
                    
                    chart_data = {
                        'ML': ml_deceptive_prob,
                        'LLM': llm_deceptive_prob,
                        'Hybrid': hybrid_score
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(chart_data)
                    
                    with col2:
                        st.metric("Agreement", "✅ Both agree" if (ml_prediction == llm_pred) else "⚠ Disagree")
                        st.metric("Risk Level", 
                            "🔴 HIGH" if hybrid_score > 0.8 else "🟡 MEDIUM" if hybrid_score > 0.5 else "🟢 LOW"
                        )
                    
                    # Show processing details if requested
                    if show_processing:
                        st.subheader("🔧 Text Processing Details")
                        st.text_area("Original text:", user_text, height=100, disabled=True)
                        st.text_area("Processed text:", cleaned_text, height=100, disabled=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if hybrid_pred == 'deceptive':
                        st.warning("""
                        *⚠ This text appears to be spam or deceptive. Consider:*
                        - Verify the source before trusting
                        - Look for unrealistic claims or urgent language
                        - Check for spelling/grammar issues
                        - Be cautious of unsolicited offers
                        - Suspicious patterns in content
                        """)
                    else:
                        st.success("""
                        *✅ This text appears to be legitimate. However:*
                        - Always use your judgment
                        - Verify important information from multiple sources
                        - Be aware that sophisticated spam can be harder to detect
                        - Consider the context and consistency
                        """)
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.error("Please enter some text to analyze!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>🛡 Hybrid Fake News Detection | ML (89%) + LLM (87%) = 88%+ Accuracy</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()