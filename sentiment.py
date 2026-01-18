import streamlit as st
from transformers import pipeline, BertForTokenClassification, BertForSequenceClassification, AutoTokenizer
import torch

# Page configuration
st.set_page_config(
    page_title="NLP Analysis Suite",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1557a0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🤖 NLP Analysis Suite</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Text Summarization & Aspect-Based Sentiment Analysis</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Select Feature")
st.sidebar.markdown("---")

feature = st.sidebar.radio(
    "Choose an analysis type:",
    ["📝 Text Summarization", "🎯 Aspect-Based Sentiment Analysis"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "**Text Summarization**: Condenses long text using BART\n\n"
    "**Aspect Sentiment**: Extracts aspects from reviews and analyzes sentiment for each aspect"
)

# Cache models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_aspect_extraction_model():
    model = BertForTokenClassification.from_pretrained("AnasAhmadz/aspect-extraction-bert")
    tokenizer = AutoTokenizer.from_pretrained("AnasAhmadz/aspect-extraction-bert")
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_sentiment_model():
    model = BertForSequenceClassification.from_pretrained("AnasAhmadz/aspect-sentiment-bert")
    tokenizer = AutoTokenizer.from_pretrained("AnasAhmadz/aspect-sentiment-bert")
    model.eval()
    return model, tokenizer

# Aspect extraction function
def extract_aspects(text, model, tokenizer):
    """Extract aspects from text using the trained BERT model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[pred.item()] for pred in predictions[0]]

    aspects = []
    current_aspect = ""
    
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if label == "B-ASP":
            if current_aspect:
                aspects.append(current_aspect.strip())
            current_aspect = token.replace("##", "")
            
        elif label == "I-ASP":
            if token.startswith("##"):
                current_aspect += token.replace("##", "")
            else:
                current_aspect += " " + token
        else:
            if current_aspect:
                aspects.append(current_aspect.strip())
                current_aspect = ""
                
    if current_aspect:
        aspects.append(current_aspect.strip())
        
    return aspects

# Sentiment analysis function
def analyze_sentiment(aspect, text, model, tokenizer):
    """Analyze sentiment for a specific aspect"""
    input_text = f"{aspect}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    sentiment = model.config.id2label[predicted_class]
    confidence = predictions[0][predicted_class].item()
    
    return sentiment, confidence

# Get color for sentiment
def get_sentiment_color(sentiment):
    colors = {
        "positive": "#28a745",
        "negative": "#dc3545",
        "neutral": "#ffc107"
    }
    return colors.get(sentiment.lower(), "#6c757d")

# Feature 1: Text Summarization
if feature == "📝 Text Summarization":
    st.markdown("## 📝 Text Summarization")
    st.markdown("Enter a long text below and get a concise summary using the BART model.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to summarize:",
            height=250,
            placeholder="Paste your long text here... (minimum 50 words recommended for best results)"
        )
    
    with col2:
        st.markdown("### ⚙️ Settings")
        max_length = st.slider("Max length:", 50, 300, 130, help="Maximum words in summary")
        min_length = st.slider("Min length:", 20, 100, 40, help="Minimum words in summary")
    
    if st.button("🚀 Generate Summary", use_container_width=True):
        if not text_input:
            st.error("⚠️ Please enter some text to summarize.")
        elif len(text_input.split()) < 20:
            st.warning("⚠️ Text is too short. Please enter at least 20 words for meaningful summarization.")
        else:
            try:
                with st.spinner("🔄 Analyzing and summarizing..."):
                    summarizer = load_summarizer()
                    summary = summarizer(
                        text_input,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                
                st.success("✅ Summary generated successfully!")
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### 📄 Summary")
                st.write(summary[0]["summary_text"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                original_words = len(text_input.split())
                summary_words = len(summary[0]["summary_text"].split())
                reduction = round((1 - summary_words/original_words) * 100, 1)
                
                with col1:
                    st.metric("📊 Original Words", original_words)
                with col2:
                    st.metric("📝 Summary Words", summary_words)
                with col3:
                    st.metric("📉 Compression", f"{reduction}%")
                    
            except Exception as e:
                st.error(f"❌ Error during summarization: {str(e)}")

# Feature 2: Aspect-Based Sentiment Analysis
elif feature == "🎯 Aspect-Based Sentiment Analysis":
    st.markdown("## 🎯 Aspect-Based Sentiment Analysis")
    st.markdown("Enter a review or feedback text. The model will identify aspects (like 'food', 'service', 'ambiance') and analyze sentiment for each.")
    
    # Example texts
    with st.expander("💡 See example reviews"):
        st.markdown("""
        **Example 1:** "The food was delicious and the atmosphere was cozy, but the service was extremely slow."
        
        **Example 2:** "Great pizza and friendly staff, but the prices are too high."
        
        **Example 3:** "The sushi was fresh and the drinks were cold. Loved the ambiance!"
        """)
    
    review_input = st.text_area(
        "Enter your review or feedback:",
        height=150,
        placeholder="Example: The food was amazing but the service was terrible and the atmosphere was too noisy."
    )
    
    if st.button("🔍 Analyze Aspects & Sentiment", use_container_width=True):
        if not review_input:
            st.error("⚠️ Please enter a review to analyze.")
        elif len(review_input.split()) < 3:
            st.warning("⚠️ Review is too short. Please enter at least a few words.")
        else:
            try:
                with st.spinner("🔄 Extracting aspects and analyzing sentiment..."):
                    # Load models
                    aspect_model, aspect_tokenizer = load_aspect_extraction_model()
                    sentiment_model, sentiment_tokenizer = load_sentiment_model()
                    
                    # Extract aspects
                    aspects = extract_aspects(review_input, aspect_model, aspect_tokenizer)
                
                if not aspects:
                    st.warning("⚠️ No specific aspects detected. Try adding more descriptive words like 'food', 'service', 'staff', etc.")
                else:
                    st.success(f"✅ Found {len(aspects)} aspect(s) in your review!")
                    
                    st.markdown("### 📊 Detailed Analysis")
                    
                    # Analyze each aspect
                    results = []
                    for aspect in aspects:
                        sentiment, confidence = analyze_sentiment(
                            aspect, review_input, sentiment_model, sentiment_tokenizer
                        )
                        results.append((aspect, sentiment, confidence))
                    
                    # Display results
                    for i, (aspect, sentiment, confidence) in enumerate(results, 1):
                        color = get_sentiment_color(sentiment)
                        
                        # Emoji based on sentiment
                        emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
                        
                        st.markdown(f"""
                        <div class="result-box" style="border-left: 5px solid {color};">
                            <h4>{emoji} Aspect {i}: <span style="color: {color};">{aspect.upper()}</span></h4>
                            <p style="margin: 0.5rem 0;">
                                <strong>Sentiment:</strong> 
                                <span style="color: {color}; font-size: 1.3rem; font-weight: bold;">
                                    {sentiment.upper()}
                                </span>
                            </p>
                            <p style="margin: 0;">
                                <strong>Confidence:</strong> {confidence:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Summary statistics
                    st.markdown("### 📈 Overall Summary")
                    
                    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
                    for _, sentiment, _ in results:
                        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                    
                    cols = st.columns(3)
                    
                    sentiments_display = [
                        ("positive", "😊 Positive", "#28a745"),
                        ("negative", "😞 Negative", "#dc3545"),
                        ("neutral", "😐 Neutral", "#ffc107")
                    ]
                    
                    for col, (sent_key, sent_label, color) in zip(cols, sentiments_display):
                        count = sentiment_counts[sent_key]
                        with col:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, {color}20, {color}10); 
                                        border-radius: 10px; border: 2px solid {color};">
                                <h2 style="color: {color}; margin: 0; font-size: 2.5rem;">{count}</h2>
                                <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: bold;">{sent_label}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Tip: Make sure your review contains descriptive words about specific aspects.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p style="margin: 0;">🤖 Built with Streamlit • 🧠 Powered by BERT & BART</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Models: <a href="https://huggingface.co/AnasAhmadz/aspect-extraction-bert" target="_blank">Aspect Extraction</a> • 
            <a href="https://huggingface.co/AnasAhmadz/aspect-sentiment-bert" target="_blank">Sentiment Analysis</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)