import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import io
from PIL import Image
import altair as alt
import os
import glob
import base64

st.set_page_config(page_title="✨ FOMO Prediction Dashboard", layout="wide")

# --- CUSTOM STYLE --- 
st.markdown(
    """
    <style>
    /* Background Gradient and General Styling */
    .stApp { 
        background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 45%, #fffbf2 100%); 
        font-family: 'Arial', sans-serif;
    }
    
    /* Main Titles */
    .big-title { 
        font-size: 36px; 
        font-weight: 800; 
        color: #0f172a;
        text-align: center;
        margin-top: 20px;
    }
    
    .subtle { 
        color: #475569;
        text-align: center;
    }

    /* Cards */
    .card { 
        background: transparent !important; 
        padding: 8px; 
        border-radius: 0; 
        box-shadow: none; 
        margin: 8px 0;
    }

    /* Metrics */
    .metric { 
        padding: 8px 12px; 
        border-radius: 8px; 
        background: #0ea5a4; 
        color: white; 
        display: inline-block;
        font-weight: 600;
    }

    /* Sliders & Inputs Styling (transparent - no card look) */
    .stSlider { 
        background-color: transparent; 
        border-radius: 4px; 
        box-shadow: none;
    }
    
    .stSelectbox, .stRadio, .stTextInput, .stNumberInput { 
        background-color: transparent !important; 
        border-radius: 0; 
        box-shadow: none !important;
    }

    /* Button Animation */
    .stButton>button { 
        background: linear-gradient(145deg, #10b981, #34d399);
        border: none; 
        border-radius: 12px;
        padding: 10px 20px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .stButton>button:hover { 
        background: linear-gradient(145deg, #34d399, #10b981); 
        transform: scale(1.05); 
        transition: all 0.3s ease;
    }

    /* Gradient Bars */
    .stProgress {
        background: linear-gradient(45deg, #10b981, #34d399);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Main Header
st.markdown('<div class="big-title">✨ FOMO Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">An interactive tool to assess and reduce your FOMO risk.</div>', unsafe_allow_html=True)

# Sidebar for Model Selection
add_selectitem = st.sidebar.selectbox("Select a model to use", ("FOMO Prediction",))

PLATFORMS = ["Instagram", "WA", "LINE", "Twitter", "Facebook", "Snapchat"]

# --- Load Model Function ---
@st.cache_resource
def load_model(path="fomo_model.pkl"):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.session_state.setdefault('model_load_error', str(e))
        return None

# --- Small helper to convert widgets into model-ready dataframe ---
def build_input_df(values):
    data = {
        "Jenis_Kelamin": 1 if values['Jenis_Kelamin'] == "Laki-laki" else 0,
        "Sering": PLATFORMS.index(values['Sering']) if values['Sering'] in PLATFORMS else 0,
        "Lupa_Waktu": PLATFORMS.index(values['Lupa_Waktu']) if values['Lupa_Waktu'] in PLATFORMS else 0,
        # derived features from sliders
        "Waktu_Harian": 1 if values['Waktu_Harian'] == "Ya" else 0,
        "Usaha_Melepaskan_Diri": 1 if values['Usaha_Melepaskan_Diri'] == "Pernah" else 0,
        "Kesulitan_Melepaskan_Diri": 1 if values['Kesulitan_Melepaskan_Diri'] == "Ya" else 0,
        "Butuh_Aplikasi": 1 if values['Butuh_Aplikasi'] == "Ya" else 0,
        "Pernah_Memakai_Aplikasi_Pengaturan_Waktu": 1 if values['Pernah_Memakai_Aplikasi_Pengaturan_Waktu'] == "Ya" else 0,
    }
    # Additional numeric simulation features stored separately
    data['Hours_Per_Day'] = values.get('Hours_Per_Day', 2)
    data['Notifications_Per_Day'] = values.get('Notifications_Per_Day', 10)
    data['Self_Control'] = values.get('Self_Control', 5)
    return pd.DataFrame(data, index=[0])

# --- Heuristic scoring function ---
def heuristic_score(df_row: pd.Series) -> float:
    h = float(df_row.get('Hours_Per_Day', 2)) / 12.0
    notif = float(df_row.get('Notifications_Per_Day', 10)) / 200.0
    self_ctrl = 1.0 - (float(df_row.get('Self_Control', 5)) / 10.0)
    difficulty = float(df_row.get('Kesulitan_Melepaskan_Diri', 0))
    daily = float(df_row.get('Waktu_Harian', 0))
    score = 0.5*h + 0.2*notif + 0.2*self_ctrl + 0.05*difficulty + 0.05*daily
    return float(np.clip(score, 0, 1))

# --- Session state storage for history ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Predict ✨", "Explore 🔎", "About ℹ️"])

# --- TAB 1: PREDICT FOMO ---
with tab1:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(180deg, #f3f4f6 0%, #ffffff 50%, #e0e7ff 100%); }
        .big-title { font-size: 36px; font-weight: 700; color: #111827; }
        .subtle { font-size: 18px; color: #475569; margin-bottom: 16px; }
        .card { background: transparent !important; padding: 8px; border-radius: 0; box-shadow: none; transition: none; }
        .card:hover { transform: scale(1.05); }
        .persona-button { font-size: 16px; background-color: #3b82f6; color: white; padding: 12px 24px; border-radius: 10px; cursor: pointer; transition: all 0.3s ease; }
        .persona-button:hover { background-color: #2563eb; }
        .prediction-result { font-size: 24px; font-weight: bold; color: #ef4444; padding: 16px; text-align: center; }
        .slider-label { font-size: 14px; font-weight: bold; color: #3b82f6; }
        .feedback { padding: 16px; font-size: 16px; border-radius: 8px; background-color: #f9fafb; margin-top: 20px; }
        .interactive-btn { font-size: 16px; background-color: #10b981; color: white; padding: 12px 24px; border-radius: 10px; cursor: pointer; transition: all 0.3s ease; }
        .interactive-btn:hover { background-color: #059669; }
        .floating-button { position: fixed; bottom: 20px; right: 20px; background-color: #3b82f6; padding: 16px 24px; border-radius: 50%; box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2); color: white; font-size: 24px; cursor: pointer; }
        .floating-button:hover { background-color: #2563eb; }
        </style>
    """, unsafe_allow_html=True)

    # Header and introduction
    st.markdown('<div class="big-title">✨ FOMO Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Predict your FOMO risk based on personalized inputs and get tips to reduce it.</p>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Predict FOMO Risk for a User")

    # Persona selection with illustrations
    st.subheader("Select a persona:")
    persona = st.selectbox('Choose a persona preset', ['Custom', 'Casual User', 'Heavy User', 'Struggling with Self-Control'])
    
    if persona == 'Casual User':
        persona_img = 'https://via.placeholder.com/150?text=Casual+User'
    elif persona == 'Heavy User':
        persona_img = 'https://via.placeholder.com/150?text=Heavy+User'
    elif persona == 'Struggling with Self-Control':
        persona_img = 'https://via.placeholder.com/150?text=Struggling+User'
    else:
        persona_img = None

    if persona_img:
        st.image(persona_img, caption=persona, width=150)

    # Persona preset logic
    if persona == 'Casual User':
        preset = dict(Jenis_Kelamin='Perempuan', Sering='Instagram', Lupa_Waktu='Instagram', Waktu_Harian='Ya', Usaha_Melepaskan_Diri='Pernah', Kesulitan_Melepaskan_Diri='Tidak', Butuh_Aplikasi='Tidak', Pernah_Memakai_Aplikasi_Pengaturan_Waktu='Tidak')
    elif persona == 'Heavy User':
        preset = dict(Jenis_Kelamin='Laki-laki', Sering='Instagram', Lupa_Waktu='Instagram', Waktu_Harian='Ya', Usaha_Melepaskan_Diri='Tidak Tidak Pernah', Kesulitan_Melepaskan_Diri='Ya', Butuh_Aplikasi='Ya', Pernah_Memakai_Aplikasi_Pengaturan_Waktu='Tidak')
    elif persona == 'Struggling with Self-Control':
        preset = dict(Jenis_Kelamin='Perempuan', Sering='Instagram', Lupa_Waktu='Twitter', Waktu_Harian='Ya', Usaha_Melepaskan_Diri='Pernah', Kesulitan_Melepaskan_Diri='Ya', Butuh_Aplikasi='Ya', Pernah_Memakai_Aplikasi_Pengaturan_Waktu='Tidak')
    else:
        preset = None

    # Input controls
    jenis_kelamin = st.radio("Jenis Kelamin", ('Perempuan', 'Laki-laki'), index=0 if (not preset or preset.get('Jenis_Kelamin', 'Perempuan') == 'Perempuan') else 1)
    sering = st.selectbox("Sering Menggunakan (utama)", PLATFORMS, index=0 if not preset else PLATFORMS.index(preset['Sering']) if preset.get('Sering') in PLATFORMS else 0)
    lupa_waktu = st.selectbox("Aplikasi yang membuat lupa waktu", PLATFORMS, index=0 if not preset else PLATFORMS.index(preset['Lupa_Waktu']) if preset.get('Lupa_Waktu') in PLATFORMS else 0)

    # Sliders with better design
    st.markdown('**Daily behavior sliders**')
    Hours_Per_Day = st.slider('Hours per day on social apps', 0.0, 12.0, 2.0, 0.5, help="How much time do you spend on social media every day?")
    Notifications_Per_Day = st.slider('Notifications per day', 0, 200, 20, help="How many notifications do you receive on average daily?")
    Self_Control = st.slider('Self-control (higher is better)', 0, 10, 6, help="How much self-control do you have regarding social media?")

    st.markdown('**Self-reported questions**')
    waktu_harian = st.radio("Apakah Menggunakan Aplikasi Setiap Hari?", ['Ya', 'Tidak'], index=0 if not preset or preset.get('Waktu_Harian', 'Ya') == 'Ya' else 1)
    usaha_melepaskan_diri = st.radio("Apakah Berusaha Melepaskan Diri dari Aplikasi?", ['Pernah', 'Tidak Tidak Pernah'], index=0)
    kesulitan_melepaskan_diri = st.radio("Apakah Mengalami Kesulitan Melepaskan Diri?", ['Ya', 'Tidak'], index=1 if not preset else 0 if preset.get('Kesulitan_Melepaskan_Diri', 'Tidak') == 'Ya' else 1)
    butuh_aplikasi = st.radio("Apakah Butuh Aplikasi Pengatur Waktu?", ['Ya', 'Tidak'], index=1)
    pernah_memakai_aplikasi = st.radio("Pernah Menggunakan Aplikasi Pengatur Waktu?", ['Ya', 'Tidak'], index=1)

    user_values = {
        'Jenis_Kelamin': jenis_kelamin,
        'Sering': sering,
        'Lupa_Waktu': lupa_waktu,
        'Waktu_Harian': waktu_harian,
        'Usaha_Melepaskan_Diri': usaha_melepaskan_diri,
        'Kesulitan_Melepaskan_Diri': kesulitan_melepaskan_diri,
        'Butuh_Aplikasi': butuh_aplikasi,
        'Pernah_Memakai_Aplikasi_Pengaturan_Waktu': pernah_memakai_aplikasi,
        'Hours_Per_Day': Hours_Per_Day,
        'Notifications_Per_Day': Notifications_Per_Day,
        'Self_Control': Self_Control
    }

    st.markdown('---')
    if st.button('Predict FOMO!'):
        # build input and attempt model
        df_input = build_input_df(user_values)
        model = load_model()

        # Progress animation
        with st.spinner('Analyzing patterns...'):
            time.sleep(0.6)
        if model is None:
            score = heuristic_score(df_input.loc[0])
        else:
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(df_input)[0]
                    score = float(prob[1])
                else:
                    pred = model.predict(df_input)[0]
                    score = float(pred)
            except Exception as e:
                st.error(f'Prediction error: {e}')
                score = heuristic_score(df_input.loc[0])

        # Prediction result
        risk_label = 'High Risk — FOMO Detected 😬' if score > 0.5 else 'Low Risk — No FOMO ✅'
        st.markdown(f"<div class='prediction-result'>{risk_label}</div>", unsafe_allow_html=True)

        # Colorful gauge chart
        gauge = pd.DataFrame({'score': [score * 100]})
        gauge_chart = alt.Chart(gauge).mark_bar(size=40).encode(
            x=alt.X('score:Q', scale=alt.Scale(domain=[0, 100])),
            color=alt.condition(alt.datum.score > 50, alt.value('#ef4444'), alt.value('#10b981'))
        )
        st.altair_chart(gauge_chart, use_container_width=True)

        # Actionable tips
        if score > 0.6:
            st.warning('Suggestion: Start with a 2-day digital detox and enable notification batching.')
        elif score > 0.4:
            st.info('Suggestion: Try app time-limits and turn off non-essential notifications.')
        else:
            st.success('Great! Maintain your healthy habits.')

        # Quick preview table of inputs and score
        st.subheader("Quick Preview 🔎")
        # Save to history
        st.session_state['history'].append({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            **user_values,
            'score': round(score, 4),
            'risk_label': risk_label
        })

        # Show history in an expander
        with st.expander("View past predictions"):
            hist_df = pd.DataFrame(st.session_state['history'])
            if not hist_df.empty:
                st.dataframe(hist_df)
            else:
                st.write("No past predictions yet.")

    # Floating action button for instant prediction
    st.markdown('<div class="floating-button">🔮</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(180deg, #f9fafb 0%, #ffffff 60%, #e7eaf0 100%); }
        .big-title { font-size:36px; font-weight:700; color: #0f172a; }
        .subtle { color:#475569; font-size: 16px; }
        .card { background: transparent !important; padding: 8px; border-radius: 0; box-shadow: none; transition: none; }
        .card:hover { transform: scale(1.05); }
        .slider-label { font-size: 14px; color: #3b82f6; font-weight: bold; }
        .slider-container { margin: 12px 0; }
        .explore-chart { margin-top: 20px; }
        .interactive-btn { font-size: 16px; background-color: #10b981; color: white; padding: 12px 24px; border-radius: 10px; cursor: pointer; transition: all 0.3s ease; }
        .interactive-btn:hover { background-color: #059669; }
        </style>
    """, unsafe_allow_html=True)

    # Header with clear title and instructions
    st.markdown('<div class="big-title">Explore What‑If Scenarios 🔎</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Experiment with how different factors (hours, notifications, self-control) influence your FOMO risk. Adjust the sliders to see real-time changes in the chart.</p>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Adjust Your Parameters Below')
    st.write("Select how much time you spend on social media, how many notifications you receive, and your self-control level. The graph will update dynamically as you change the parameters.")
    
    # Interactive Sliders
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fixed_notifications = st.slider('Notifications per day (fixed during simulation)', 0, 200, 20, key="notif", help="How many notifications do you receive on average each day?")
        fixed_self_control = st.slider('Self-control (fixed)', 0, 10, 6, key="control", help="How easy is it for you to control your usage?")
        fixed_daily_use = st.selectbox('Do you use apps daily?', ['Ya', 'Tidak'], index=0, key="daily_use", help="Do you use social media apps every day?")
        fixed_difficulty = st.selectbox('Difficulty to stop using apps?', ['Ya', 'Tidak'], index=1, key="difficulty", help="Do you find it hard to stop using apps once you start?")

    # Simulating the impact of these parameters on FOMO risk
    hours = np.linspace(0, 12, 25)
    scores = []
    for h in hours:
        row = {
            'Hours_Per_Day': h, 
            'Notifications_Per_Day': fixed_notifications, 
            'Self_Control': fixed_self_control, 
            'Waktu_Harian': 1 if fixed_daily_use == 'Ya' else 0, 
            'Kesulitan_Melepaskan_Diri': 1 if fixed_difficulty == 'Ya' else 0
        }
        scores.append(heuristic_score(pd.Series(row)))
    
    sim_df = pd.DataFrame({'hours': hours, 'score': scores})

    # Dynamic interactive chart using Altair
    line = alt.Chart(sim_df).mark_line(point=True, color='#3b82f6').encode(
        x='hours', 
        y=alt.Y('score', scale=alt.Scale(domain=[0, 1])), 
        tooltip=['hours', 'score']
    ).properties(width=600, height=350).interactive()

    # Show the chart
    st.altair_chart(line, use_container_width=True, key="fomo_chart")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive Button for personalized recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write('Want personalized recommendations for reducing FOMO? Click below to get tips tailored to your data.')
    
    if st.button('Generate Recommendations', key="recommendations", help="Get personalized suggestions for reducing your FOMO based on your inputs."):
        if fixed_self_control < 5:
            st.success("✅ **Recommendation:** Improve self-control! Start with a 2-day digital detox and limit notifications to essential apps.")
        elif fixed_notifications > 100:
            st.info("⚠️ **Recommendation:** You may want to reduce notifications. Try enabling notification batching for non-urgent apps.")
        else:
            st.success("🎉 **Recommendation:** You're on the right track! Keep up with healthy digital habits and try limiting app usage to 2-3 hours per day.")
        
        st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)

    # Conclusion with an encouraging note
    st.markdown("""
        <div class="card" style="background-color:#eef2f7;">
            <p class="subtle">Remember: Small, consistent changes in your digital habits can have a big impact on your well-being. Explore different scenarios and find the best balance for you!</p>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    # Adding a smooth gradient background and card design
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(180deg, #f0f4f8 0%, #ffffff 60%, #e7eaf0 100%); }
        .big-title { font-size:36px; font-weight:700; color: #0f172a; }
        .subtle { font-size:16px; color:#475569; }
        .card { background: transparent !important; padding: 8px; border-radius: 0; box-shadow: none; transition: none; }
        .card:hover { transform: scale(1.05); }
        .icon { width: 30px; height: 30px; margin-right: 10px; vertical-align: middle; }
        .tips-list { list-style-type: none; padding-left: 0; }
        .tips-list li { display: flex; align-items: center; padding: 8px 0; }
        .download-btn { font-weight: bold; font-size: 16px; color: white; background-color: #3b82f6; padding: 10px 24px; border-radius: 12px; }
        .download-btn:hover { background-color: #2563eb; cursor: pointer; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with enhanced visual styling
    st.markdown('<div class="big-title">About this App ✨</div>', unsafe_allow_html=True)
    st.markdown("<p class='subtle'>Welcome to the FOMO Prediction Dashboard — a tool that helps you understand your digital habits and predict your risk of FOMO (Fear of Missing Out). Let's dive in!</p>", unsafe_allow_html=True)

    # Image with dynamic fade-in effect
    st.image('self-care-4899284_640.jpg', caption='Stay connected while maintaining your mental well-being', width=900)
    
    # Brief explanation of the app with an interactive card design
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("This app uses advanced machine learning models to predict how likely you are to experience FOMO based on several behavioral inputs like:")
    st.markdown("""
        - **Time spent on social media**  
        - **Notifications you receive**  
        - **Your ability to control usage**
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips to reduce FOMO with icons for better UX
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Tips to Reduce FOMO:')
    
    tips = [
        {"text": "📴 Digital Detox: Take breaks from screens. Schedule phone-free times, especially during meals and before bed."},
        {"text": "🔕 Turn off non-essential notifications: Reduce interruptions and focus on important tasks."},
        {"text": "⏰ Set app time limits: Limit your time spent on apps to avoid mindless scrolling."},
        {"text": "🧘 Mindful usage: Be conscious about your digital habits. Try activities that don't involve screens."},
        {"text": "📱 Use time management apps: Download apps to monitor your usage and block distracting notifications."}
    ]
    
    st.markdown('<ul class="tips-list">', unsafe_allow_html=True)
    for tip in tips:
        st.markdown(f'<li>{tip["text"]}</li>', unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # If there’s a model load error, show a warning with a red card background
    if 'model_load_error' in st.session_state:
        st.markdown('<div class="card" style="background-color:#fcd34d;">', unsafe_allow_html=True)
        st.warning(f"⚠️ **Model Load Error:** {st.session_state['model_load_error']}", icon="🚨")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download sample dataset with more interactive button design
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write('Test the model with your own data or download the sample dataset below:')
    
    sample = pd.DataFrame([{
        'Jenis_Kelamin': 0, 'Sering': 0, 'Lupa_Waktu': 0, 'Waktu_Harian': 1, 'Usaha_Melepaskan_Diri': 0, 
        'Kesulitan_Melepaskan_Diri': 0, 'Butuh_Aplikasi': 1, 'Pernah_Memakai_Aplikasi_Pengaturan_Waktu': 0
    }, {
        'Jenis_Kelamin': 1, 'Sering': 2, 'Lupa_Waktu': 2, 'Waktu_Harian': 1, 'Usaha_Melepaskan_Diri': 1, 
        'Kesulitan_Melepaskan_Diri': 1, 'Butuh_Aplikasi': 1, 'Pernah_Memakai_Aplikasi_Pengaturan_Waktu': 1
    }])
    
    st.download_button('Download Sample CSV', data=sample.to_csv(index=False).encode('utf-8'), file_name='fomo_example.csv', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # About the Team section — show local group photo if available and member intros
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('About the Team ✨')
    st.write("We’re a small team of researchers, designers, and engineers who built this dashboard. If you have images in this project folder they will be shown automatically.")

    # Auto-detect local image files in the current working directory
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.gif'):
        image_paths.extend(glob.glob(os.path.join(os.getcwd(), ext)))
    image_paths = sorted(image_paths)

    if image_paths:
        group_photo_local = image_paths[1]
        def _img_data_uri(p):
            try:
                with open(p, 'rb') as f:
                    b = f.read()
                mime = 'image/jpeg'
                if p.lower().endswith('.png'):
                    mime = 'image/png'
                elif p.lower().endswith('.gif'):
                    mime = 'image/gif'
                return f"data:{mime};base64,{base64.b64encode(b).decode()}"
            except Exception:
                return None

        src = _img_data_uri(group_photo_local)
        if src:
            st.markdown(f"<div style='text-align:center'><img src='{src}' alt='Team photo' style='max-width:90%; height:auto; border-radius:8px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center'><img src='https://via.placeholder.com/900x320?text=Team+Photo' alt='Team' style='max-width:90%; height:auto; border-radius:8px;'></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center'><img src='https://via.placeholder.com/900x320?text=Team+Photo' alt='Team' style='max-width:90%; height:auto; border-radius:8px;'></div>", unsafe_allow_html=True)

    # Team members (5 people). For each member, if a local image name contains the member name it will be used.
    team = [
        {"name": "Elviraasanti Chaerunnisa", "role": "Lead Developer", "photo": "https://via.placeholder.com/150?text=Aliif", "bio": "Frontend & integration lead."},
        {"name": "Ghina Alifiyah", "role": "Data Scientist", "photo": "https://via.placeholder.com/150?text=Alex", "bio": "Modeling & analysis."},
        {"name": "Yonanda Rianita", "role": "Product Designer", "photo": "https://via.placeholder.com/150?text=Sam", "bio": "UX & visual design."},
        {"name": "Putri Patricia", "role": "Researcher", "photo": "https://via.placeholder.com/150?text=Taylor", "bio": "User research & evaluation."},
        {"name": "Muhamamd Alif Fahrezi", "role": "Engineer", "photo": "https://via.placeholder.com/150?text=Jordan", "bio": "Backend & infra."}
    ]

    # Try matching local files to member names (case-insensitive)
    for member in team:
        name_key = member['name'].lower()
        matched = None
        for p in image_paths:
            if name_key in os.path.basename(p).lower():
                matched = p
                break
        if matched:
            member['photo_local'] = matched
        else:
            member['photo_local'] = member['photo']

    cols = st.columns(len(team))
    for c, member in zip(cols, team):
        img_src = member.get('photo_local', member['photo'])
        try:
            c.markdown(f"<div style='text-align:center'><img src='{img_src}' width='140' style='border-radius:8px;'></div>", unsafe_allow_html=True)
        except Exception:
            c.markdown("<div style='text-align:center'><img src='https://via.placeholder.com/150?text=No+Image' width='140'></div>", unsafe_allow_html=True)
        c.markdown(f"<p style='text-align:center'><strong>{member['name']}</strong><br/>{member['role']}<br/>{member['bio']}</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="background-color:#eef2f7;">', unsafe_allow_html=True)
    st.caption('Made with ❤️ by the FOMO Prediction Team. Experiment with different inputs and enjoy exploring your digital habits!')
    st.markdown('</div>', unsafe_allow_html=True)
