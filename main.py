import gradio as gr
from transformers import pipeline
import json
import time
import traceback

# Global variable to store classifier
classifier = None

def load_model():
    """Load the model with proper error handling"""
    global classifier
    try:
        print("Loading model from Hugging Face...")
        classifier = pipeline(
            "text-classification", 
            model="mgulati3/news-classifier-model",
            return_all_scores=True  # This ensures we get all category scores
        )
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        return False

def classify_with_animation(text):
    """Enhanced classification with proper error handling"""
    global classifier
    
    # Check if text is empty
    if not text or not text.strip():
        return {}, "Please enter some text to classify."
    
    # Check if model is loaded
    if classifier is None:
        model_loaded = load_model()
        if not model_loaded:
            return {}, "❌ **Model Loading Error**\n\nFailed to load the classification model. Please check your internet connection and try again."
    
    try:
        print(f"Classifying text: {text[:100]}...")
        
        # Add a small delay for demo effect
        time.sleep(0.5)
        
        # Get predictions
        results = classifier(text)
        print(f"Raw results: {results}")
        
        # Handle different result formats
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                # If results is a list of lists, take the first list
                predictions = results[0]
            else:
                # If results is a list of dicts
                predictions = results
        else:
            predictions = results
        
        # Format results for Gradio Label component
        formatted_results = {}
        for item in predictions:
            if isinstance(item, dict) and 'label' in item and 'score' in item:
                formatted_results[item['label']] = round(item['score'], 4)
        
        if not formatted_results:
            return {}, "❌ **Classification Error**\n\nNo valid predictions returned from the model."
        
        # Create a detailed breakdown with emojis for visual appeal
        top_category = max(predictions, key=lambda x: x['score'])
        confidence_level = "🎯 High" if top_category['score'] > 0.7 else "⚡ Medium" if top_category['score'] > 0.4 else "📊 Low"
        
        # Category emojis for visual appeal
        category_emojis = {
            'politics': '🏛️',
            'health': '🏥',
            'business': '💼',
            'science': '🔬',
            'climate': '🌱',
            'POLITICS': '🏛️',
            'HEALTH': '🏥',
            'BUSINESS': '💼',
            'SCIENCE': '🔬',
            'CLIMATE': '🌱'
        }
        
        emoji = category_emojis.get(top_category['label'], '📰')
        
        summary = f"## {emoji} **{top_category['label'].title()}** Category Detected\n\n"
        summary += f"**Confidence Level:** {confidence_level} *({top_category['score']:.1%})*\n\n"
        summary += f"**Model Performance:** Analyzed {len(predictions)} categories\n\n"
        summary += f"**🤖 AI Model:** Fine-tuned on Hugging Face\n\n"
        summary += f"**📊 All Scores:**\n"
        for item in sorted(predictions, key=lambda x: x['score'], reverse=True):
            emoji_cat = category_emojis.get(item['label'], '📰')
            summary += f"- {emoji_cat} {item['label'].title()}: {item['score']:.1%}\n"
        
        return formatted_results, summary
        
    except Exception as e:
        error_msg = f"❌ **Classification Error**\n\nError: {str(e)}\n\nPlease try again or check the model configuration."
        print(f"Classification error: {str(e)}")
        print(traceback.format_exc())
        return {}, error_msg

def test_model():
    """Test the model with a sample text"""
    test_text = "The Federal Reserve announced a new interest rate policy."
    try:
        result = classify_with_animation(test_text)
        print(f"Model test successful: {result}")
        return True
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False

# Enhanced CSS for video-ready UI
custom_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main container styling */
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    border-radius: 25px !important;
    padding: 40px !important;
    box-shadow: 0 25px 50px rgba(0,0,0,0.15) !important;
    font-family: 'Inter', sans-serif !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Animated background effect */
.gradio-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
    animation: shimmer 3s infinite;
    pointer-events: none;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Header styling */
.header {
    text-align: center;
    margin-bottom: 35px;
    position: relative;
    z-index: 10;
}

/* Title styling with glow effect */
h1 {
    color: white !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin-bottom: 15px !important;
    text-shadow: 0 0 20px rgba(255,255,255,0.5) !important;
    animation: glow 2s ease-in-out infinite alternate !important;
}

@keyframes glow {
    from { text-shadow: 0 0 20px rgba(255,255,255,0.5); }
    to { text-shadow: 0 0 30px rgba(255,255,255,0.8), 0 0 40px rgba(255,255,255,0.3); }
}

/* Description styling */
.description {
    color: #f8fafc !important;
    font-size: 1.2rem !important;
    margin-bottom: 25px !important;
    font-weight: 300 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
}

/* Input/Output container styling */
.input-output-container {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(25px) !important;
    border-radius: 20px !important;
    padding: 30px !important;
    margin: 25px 0 !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    position: relative !important;
    z-index: 10 !important;
}

/* Textbox styling */
.gr-textbox textarea {
    border: 2px solid #e2e8f0 !important;
    border-radius: 15px !important;
    background: #fafafa !important;
    font-size: 1.1rem !important;
    padding: 20px !important;
    transition: all 0.4s ease !important;
    color: #1f2937 !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6 !important;
}

.gr-textbox textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
    background: white !important;
    transform: translateY(-2px) !important;
}

/* Label styling */
.gr-textbox label, .gr-label label {
    color: #374151 !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
    font-size: 1.1rem !important;
}

/* Button styling with hover animations */
.gr-button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 15px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 15px 30px !important;
    transition: all 0.4s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 1rem !important;
    position: relative !important;
    overflow: hidden !important;
}

.gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 25px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:active {
    transform: translateY(-1px) !important;
}

/* Label output styling */
.gr-label {
    background: #f8fafc !important;
    border-radius: 15px !important;
    padding: 25px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
}

.gr-label .output-class {
    color: #1f2937 !important;
    font-weight: 600 !important;
}

/* Markdown output styling */
.gr-markdown {
    color: #374151 !important;
    line-height: 1.7 !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #1f2937 !important;
    margin-bottom: 12px !important;
}

.gr-markdown strong {
    color: #667eea !important;
}

/* General text styling */
.gr-block label {
    color: #374151 !important;
    font-weight: 600 !important;
}

/* Examples styling */
.gr-examples {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin: 25px 0 !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    backdrop-filter: blur(10px) !important;
}

.gr-examples .gr-button {
    background: #f8fafc !important;
    color: #475569 !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    margin: 8px !important;
    padding: 12px 20px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    transition: all 0.3s ease !important;
}

.gr-examples .gr-button:hover {
    background: #e2e8f0 !important;
    border-color: #667eea !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px rgba(0,0,0,0.1) !important;
}

/* Info box styling */
.info-box {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 20px 0 !important;
    border-left: 5px solid #667eea !important;
    color: #374151 !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1) !important;
    backdrop-filter: blur(10px) !important;
}

.info-box strong {
    color: #1f2937 !important;
}

/* Category badge styling */
.category-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    padding: 8px 16px;
    border-radius: 25px;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
}

.category-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
}

/* Section headers */
.section-header {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    margin-bottom: 20px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
    font-size: 1.3rem !important;
}

/* Professional badge */
.professional-badge {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    padding: 8px 20px !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    display: inline-block !important;
    margin: 10px 0 !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 25px !important;
        margin: 15px !important;
    }
    
    h1 {
        font-size: 2.2rem !important;
    }
    
    .input-output-container {
        padding: 20px !important;
    }
    
    .category-badge {
        font-size: 0.85rem !important;
        padding: 6px 12px !important;
    }
}

/* Loading animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 1.5s infinite;
}
"""

# Load model on startup
print("Initializing News Classifier...")
model_loaded = load_model()

if not model_loaded:
    print("WARNING: Model failed to load. The app will start but classification won't work.")

# Create the enhanced interface
with gr.Blocks(
    css=custom_css,
    title="🚀 AI News Classifier - Professional Demo",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
    )
) as demo:
    
    # Header with professional branding
    gr.HTML("""
    <div class="header">
        <h1>🚀 AI News Category Classifier</h1>
        <p class="description">Advanced Machine Learning Model | Fine-tuned on Hugging Face</p>
        <div class="professional-badge">✨ Custom Fine-tuned LLM</div>
    </div>
    """)
    
    # Enhanced info box
    gr.HTML("""
    <div class="info-box">
        <strong>🎯 AI-Powered Classification System</strong><br>
        <p style="margin-top: 10px; font-size: 1rem; line-height: 1.6;">
            This model has been specifically fine-tuned to classify news articles into 5 distinct categories with high accuracy:
        </p>
        <div style="margin-top: 15px;">
            <span class="category-badge">🏛️ Politics</span>
            <span class="category-badge">🏥 Health</span>
            <span class="category-badge">💼 Business</span>
            <span class="category-badge">🔬 Science</span>
            <span class="category-badge">🌱 Climate</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.HTML('<div class="input-output-container">')
            
            text_input = gr.Textbox(
                lines=7,
                placeholder="📝 Enter your news article text here...\n\nExample: 'Researchers at MIT have developed a revolutionary AI system that can predict climate patterns with 95% accuracy, potentially transforming how we understand and respond to climate change.'",
                label="📰 News Article Content",
                info="🔍 Paste any news article text for instant AI-powered categorization"
            )
            
            with gr.Row():
                classify_btn = gr.Button("🚀 Analyze with AI", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear Text", variant="secondary", scale=1)
            
            gr.HTML('</div>')
    
    with gr.Row():
        with gr.Column(scale=1):
            # Output section
            gr.HTML('<div class="input-output-container">')
            
            label_output = gr.Label(
                label="🎯 AI Classification Results",
                num_top_classes=5
            )
            
            summary_output = gr.Markdown(
                label="📊 Detailed Analysis"
            )
            
            gr.HTML('</div>')
    
    # Examples section with better presentation
    gr.HTML("""
    <div style="margin: 25px 0;">
        <h3 class="section-header">💡 Demo Examples - Click to Try:</h3>
    </div>
    """)
    
    examples = gr.Examples(
        examples=[
            ["Scientists at Stanford University have developed a groundbreaking AI algorithm that can detect early signs of Alzheimer's disease through simple eye scans, potentially revolutionizing early diagnosis and treatment approaches."],
            ["The Federal Reserve announced a surprise interest rate cut of 0.5%, sending tech stocks soaring as investors anticipate increased liquidity in the market and renewed growth in the technology sector."],
            ["Climate researchers report that Arctic sea ice has reached its second-lowest extent on record, with temperatures in the region rising at twice the global average, accelerating concerns about global warming impacts."],
            ["Congressional leaders from both parties have reached a bipartisan agreement on the new infrastructure bill, allocating $2 trillion for roads, bridges, and broadband expansion across all 50 states."],
            ["Medical breakthrough: Gene therapy trial shows 90% success rate in treating previously incurable genetic blindness, offering hope to millions of patients worldwide affected by inherited retinal diseases."]
        ],
        inputs=text_input,
        outputs=[label_output, summary_output],
        fn=classify_with_animation,
        cache_examples=False,  # Disable caching to avoid errors
        label="🎬 Professional Demo Examples:"
    )
    
    # Event handlers
    classify_btn.click(
        fn=classify_with_animation,
        inputs=text_input,
        outputs=[label_output, summary_output],
        show_progress=True
    )
    
    clear_btn.click(
        fn=lambda: ("", {}, ""),
        inputs=None,
        outputs=[text_input, label_output, summary_output]
    )
    
    text_input.submit(
        fn=classify_with_animation,
        inputs=text_input,
        outputs=[label_output, summary_output],
        show_progress=True
    )
    
    # Professional footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 40px; padding: 25px; 
                background: rgba(255,255,255,0.1); border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.2);">
        <p style="color: #f1f5f9; font-size: 1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); margin-bottom: 10px;">
            🤖 <strong>Custom Fine-tuned Model</strong> | Powered by Hugging Face Transformers
        </p>
        <p style="color: #e2e8f0; font-size: 0.9rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            🚀 Built with Advanced Machine Learning | Ready for Production
        </p>
    </div>
    """)

# Launch configuration for demo
if __name__ == "__main__":
    print(f"Model loaded: {model_loaded}")
    if model_loaded:
        print("Testing model...")
        test_model()
    
    print("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,  # Enable debug to see errors
        quiet=False  # Show console output for debugging
    )