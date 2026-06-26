import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Smart Menu Classifier System",
    page_icon="🤖",
    layout="wide"
)

# --- 2. Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #eee;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Sidebar Information ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046747.png", width=100)
    st.title("Project Details")
    st.info("""
    **Core Engine:** ResNet50  
    **API:** FastAPI (Backend)  
    **Interface:** Streamlit  
    
    **Supported Classes:**
    - 🍹 Drink  
    - 🍔 Food  
    - 🪑 Interior  
    - 📄 Menu  
    - 🏠 Outside
    """)
    st.sidebar.write("---")
st.sidebar.markdown(f"""
    <div style='color: #0056b3; font-size: 0.85rem; line-height: 1.4;'>
        Developed for: <br>
        <b>Smart Menu Classifier System 2026</b><br>
        by   <span style='color: #ffffff;'>&nbsp; Omar Khalil</span>
    </div>
""", unsafe_allow_html=True)

# --- 4. Main UI Header ---
st.markdown("<h1 style='color: #0056b3;'>🍽️ Smart Menu Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #0056b3;'>Leveraging Deep Learning to categorize restaurant-related images instantly.</p>", unsafe_allow_html=True)

# --- 5. Layout Columns ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h3 style='color: #0056b3; font-weight: bold;'>📤 Image Upload</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #0056b3;'>Drop your image here (JPG, PNG)....</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your image here (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Preview', use_column_width=True)
        analyze_btn = st.button('Analyze Image 🔍')

with col2:
    st.markdown("<h3 style='color: #0056b3; font-family: sans-serif;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
    
    if uploaded_file and analyze_btn:
        with st.spinner('AI Brain is processing...'):
            files = {"file": uploaded_file.getvalue()}
            try:
                # API Call to FastAPI
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                res = response.json()
                
                # Display Result Metric
                confidence_score = res['confidence'] * 100
                st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style='color: #6c757d; margin-bottom: 5px;'>Top Prediction</h4>
                        <h1 style='color: #007BFF; margin-top: 0; font-size: 45px;'>{res['prediction']}</h1>
                        <h3 style='color: #28a745;'>{confidence_score:.2f}% Confidence</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- Visualizing Probabilities with High Contrast Plotly Chart ---
                probs = res['all_probs']
                sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1]))
                
                LABEL_COLOR = "#212529"
                GRID_COLOR = "#dee2e6"

                fig = go.Figure(go.Bar(
                    x=list(sorted_probs.values()),
                    y=list(sorted_probs.keys()),
                    orientation='h',
                    marker=dict(
                        color='rgba(0, 123, 255, 0.85)', 
                        line=dict(color='#0056b3', width=2)
                    ),
                    text=[f"<b>{v*100:.1f}%</b>" for v in sorted_probs.values()],
                    textposition='inside',
                    insidetextfont=dict(color='white', size=14),
                ))
                
                fig.update_layout(
                    title=dict(
                        text="<b>Probability Distribution per Category</b>",
                        font=dict(size=20, color=LABEL_COLOR),
                        x=0.5
                    ),
                    xaxis=dict(
                        title="<b>Confidence Score</b>",
                        titlefont=dict(size=14, color=LABEL_COLOR),
                        tickfont=dict(size=12, color=LABEL_COLOR, family="Arial Bold"),
                        tickformat='.0%',
                        range=[0, 1.1],
                        gridcolor=GRID_COLOR,
                        zerolinecolor=GRID_COLOR
                    ),
                    yaxis=dict(
                        title="<b>Category</b>",
                        titlefont=dict(size=14, color=LABEL_COLOR),
                        tickfont=dict(size=14, color=LABEL_COLOR, family="Arial Bold"),
                    ),
                    height=450,
                    margin=dict(l=20, r=40, t=70, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.5)',
                    hovermode="y unified"
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            except Exception as e:
                st.error(f"Backend connection failed. Is FastAPI running on port 8000? \nError: {e}")
    else:
        st.markdown(f"<div style='color: #0056b3; background-color: #f0f7ff; padding: 10px; border-radius: 5px;'>Upload an image and click 'Analyze' to see the AI prediction.</div>", unsafe_allow_html=True)
        
# --- Footer ---
st.write("---")
st.markdown("<p style='color: #0056b3; font-size: 0.8rem; text-align: center; opacity: 0.8;'>© 2026 Smart Menu Classifier | Enhanced High-Contrast UI</p>", unsafe_allow_html=True)