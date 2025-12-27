import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import io
from PIL import Image
import altair as alt

st.set_page_config(page_title="✨ FOMO Prediction Dashboard", layout="wide")

# --- CUSTOM STYLE ---
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 45%, #fffbf2 100%);} 
    .big-title { font-size:34px; font-weight:700; color: #0f172a; }
    .subtle { color:#475569 }
    .card { background: linear-gradient(90deg,#ffffffaa,#f8fafc); padding:12px; border-radius:12px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
    .metric { padding:8px 12px; border-radius:8px; background: #0ea5a4; color: white; display:inline-block}
    </style>
    """,
    unsafe_allow_html=True,
)

# Identitas kelompok
st.markdown('<div class="big-title">Kelompok: </div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">1. ELVIRASANTI CHAERUNNISA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">2. GHINA ALIFIYAH</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">3. YONANDA RIANITA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">4. PUTRI PATRICIA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">5. MUHAMMAD ALIF FAHREZI</div>', unsafe_allow_html=True)

st.markdown('<div class="big-title">✨ FOMO Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Beautiful interactive UI with personas, what‑if analysis, and quick tips to reduce FOMO.</div>', unsafe_allow_html=True)

add_selectitem = st.sidebar.selectbox("Select a model to use", ("FOMO Prediction",))

PLATFORMS = ["Instagram", "WA", "LINE", "Twitter", "Facebook", "Snapchat"]

# --- Load model function ---
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

# --- Heuristic scoring function (smooth, interpretable) ---
def heuristic_score(df_row: pd.Series) -> float:
    # Hours: 0-12 scaled
    h = float(df_row.get('Hours_Per_Day', 2)) / 12.0
    notif = float(df_row.get('Notifications_Per_Day', 10)) / 200.0
    self_ctrl = 1.0 - (float(df_row.get('Self_Control', 5)) / 10.0)
    difficulty = float(df_row.get('Kesulitan_Melepaskan_Diri', 0))
    daily = float(df_row.get('Waktu_Harian', 0))
    score = 0.5*h + 0.2*notif + 0.2*self_ctrl + 0.05*difficulty + 0.05*daily
    return float(np.clip(score, 0, 1))

# --- Add session state storage for history ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Predict ✨", "Explore 🔎", "About ℹ️"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    left, right = st.columns([2,1])
    with left:
        st.header('Predict FOMO for a user')

        # Persona presets
        persona = st.selectbox('Choose a persona preset', ['Custom', 'Casual User', 'Heavy User', 'Struggling with Self-Control'])

        if persona == 'Casual User':
            preset = dict(Jenis_Kelamin='Perempuan', Sering='Instagram', Lupa_Waktu='Instagram', Waktu_Harian='Ya', Usaha_Melepaskan_Diri='Pernah', Kesulitan_Melepaskan_Diri='Tidak', Butuh_Aplikasi='Tidak', Pernah_Memakai_Aplikasi_Pengaturan_Waktu='Tidak')
        elif persona == 'Heavy User':
            preset = dict(Jenis_Kelamin='Laki-laki', Sering='Instagram', Lupa_Waktu='Instagram', Waktu_Harian='Ya', Usaha_Melepaskan_Diri='Tidak Tidak Pernah', Kesulitan_Melepaskan_Diri='Ya', Butuh_Aplikasi='Ya', Pernah_Memakai_Aplikasi_Pengaturan_Waktu='Tidak')
        elif persona == 'Struggling with Self-Control':
            preset = dict(Jenis_Kelamin='Perempuan', Sering='Instagram', Lupa_Waktu='Twitter', Waktu_Harian='Ya', Usaha_Melepaskan_Diri='Pernah', Kesulitan_Melepaskan_Diri='Ya', Butuh_Aplikasi='Ya', Pernah_Memakai_Aplikasi_Pengaturan_Waktu='Tidak')
        else:
            preset = None

        # Input controls
        jenis_kelamin = st.radio("Jenis Kelamin", ('Perempuan', 'Laki-laki'), index=0 if (not preset or preset.get('Jenis_Kelamin','Perempuan')=='Perempuan') else 1)
        sering = st.selectbox("Sering Menggunakan (utama)", PLATFORMS, index=0 if not preset else PLATFORMS.index(preset['Sering']) if preset.get('Sering') in PLATFORMS else 0)
        lupa_waktu = st.selectbox("Aplikasi yang membuat lupa waktu", PLATFORMS, index=0 if not preset else PLATFORMS.index(preset['Lupa_Waktu']) if preset.get('Lupa_Waktu') in PLATFORMS else 0)

        st.markdown('**Daily behavior sliders**')
        Hours_Per_Day = st.slider('Hours per day on social apps', 0.0, 12.0, 2.0, 0.5)
        Notifications_Per_Day = st.slider('Notifications per day', 0, 200, 20)
        Self_Control = st.slider('Self-control (higher is better)', 0, 10, 6)

        st.markdown('**Self-reported questions**')
        waktu_harian = st.radio("Apakah Menggunakan Aplikasi Setiap Hari?", ['Ya', 'Tidak'], index=0 if not preset or preset.get('Waktu_Harian','Ya')=='Ya' else 1)
        usaha_melepaskan_diri = st.radio("Apakah Berusaha Melepaskan Diri dari Aplikasi?", ['Pernah', 'Tidak Tidak Pernah'], index=0)
        kesulitan_melepaskan_diri = st.radio("Apakah Mengalami Kesulitan Melepaskan Diri?", ['Ya', 'Tidak'], index=1 if not preset else 0 if preset.get('Kesulitan_Melepaskan_Diri','Tidak')=='Ya' else 1)
        butuh_aplikasi = st.radio("Apakah Butuh Aplikasi Pengatur Waktu?", ['Ya', 'Tidak'], index=1)
        pernah_memakai_aplikasi = st.radio("Pernah Menggunakan Aplikasi Pengatur Waktu?", ['Ya', 'Tidak'], index=1)

        user_values = {'Jenis_Kelamin':jenis_kelamin,'Sering':sering,'Lupa_Waktu':lupa_waktu,'Waktu_Harian':waktu_harian,'Usaha_Melepaskan_Diri':usaha_melepaskan_diri,'Kesulitan_Melepaskan_Diri':kesulitan_melepaskan_diri,'Butuh_Aplikasi':butuh_aplikasi,'Pernah_Memakai_Aplikasi_Pengaturan_Waktu':pernah_memakai_aplikasi,'Hours_Per_Day':Hours_Per_Day,'Notifications_Per_Day':Notifications_Per_Day,'Self_Control':Self_Control}

        st.markdown('---')
        if st.button('Predict FOMO!'):
            # build input and attempt model
            df_input = build_input_df(user_values)
            model = load_model()

            # progress animation
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

            risk_label = 'High Risk — FOMO Detected 😬' if score > 0.5 else 'Low Risk — No FOMO ✅'
            st.markdown(f"<div class='card'><h3 style='margin:6px'>{risk_label}</h3><p class='subtle'>Confidence: {score:.2f}</p></div>", unsafe_allow_html=True)

            # colorful gauge-like bar
            gauge = pd.DataFrame({'score': [score * 100]})
            gauge_chart = alt.Chart(gauge).mark_bar(size=40).encode(x=alt.X('score:Q', scale=alt.Scale(domain=[0, 100])), color=alt.condition(alt.datum.score > 50, alt.value('#ef4444'), alt.value('#10b981')))
            st.altair_chart(gauge_chart, use_container_width=True)

            # actionable tip
            if score > 0.6:
                st.warning('Suggestion: Start with a 2-day digital detox and enable notification batching.')
            elif score > 0.4:
                st.info('Suggestion: Try app time-limits and turn off non-essential notifications.')
            else:
                st.success('Great! Maintain your healthy habits.')

            # record history
            st.session_state.history.append({'time': time.asctime(), 'score': score, 'label': risk_label, 'hours': Hours_Per_Day, 'notifications': Notifications_Per_Day})

    with right:
        st.header('Quick preview')
        st.write('Adjust sliders and options to see sandboxed results.')
        st.write('Recent Predictions:')
        hist = st.session_state.get('history', [])
        if hist:
            st.table(pd.DataFrame(hist).tail(5))
        else:
            st.write('No predictions yet — try one!')
        st.markdown('---')
        st.image('https://images.unsplash.com/photo-1588702547923-7093a6c3ba33?auto=format&fit=crop&w=800&q=60', caption='Digital wellbeing', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header('What‑If Analysis')
    st.write('Explore how hours, notifications, and self-control affect the FOMO risk.')

    base_controls_col, plot_col = st.columns([1, 2])
    with base_controls_col:
        fixed_notifications = st.slider('Notifications (fixed during simulation)', 0, 200, 20)
        fixed_self_control = st.slider('Self-control (fixed)', 0, 10, 6)
        fixed_daily_use = st.selectbox('Daily use?', ['Ya', 'Tidak'], index=0)
        fixed_difficulty = st.selectbox('Difficulty to stop using?', ['Ya', 'Tidak'], index=1)

    # simulate across hours 0..12
    hours = np.linspace(0, 12, 25)
    scores = []
    for h in hours:
        row = {'Hours_Per_Day': h, 'Notifications_Per_Day': fixed_notifications, 'Self_Control': fixed_self_control, 'Waktu_Harian': 1 if fixed_daily_use == 'Ya' else 0, 'Kesulitan_Melepaskan_Diri': 1 if fixed_difficulty == 'Ya' else 0}
        scores.append(heuristic_score(pd.Series(row)))
    sim_df = pd.DataFrame({'hours': hours, 'score': scores})

    line = alt.Chart(sim_df).mark_line(point=True).encode(x='hours', y=alt.Y('score', scale=alt.Scale(domain=[0, 1])), tooltip=['hours', 'score'])
    st.altair_chart(line.interactive(), use_container_width=True)

    st.markdown('**Try dragging the dots or changing fixed sliders to immediately see changes.**')

    if st.button('Generate recommendations for reducing FOMO'):
        st.success('Recommendation: Reduce hours by 20-30%, disable non-essential notifications, and schedule designated "no-phone" periods.')
        st.balloons()

with tab3:
    st.header('About this app')
    st.write('This app demonstrates an interactive UI for exploring FOMO risk with sliders, personas, and what-if plots. It uses a model when available and a clear heuristic fallback otherwise.')
    if 'model_load_error' in st.session_state:
        st.warning('Model could not be loaded: ' + st.session_state['model_load_error'])
    st.markdown('---')
    st.write('Tips to reduce FOMO:')
    st.markdown('- Turn off notifications for non-essential apps\n- Schedule phone-free blocks (e.g., during meals)\n- Use app time limits and check-ins to reduce impulsive use')
    st.write('Download a sample dataset to test batch predictions:')
    sample = pd.DataFrame([{
        'Jenis_Kelamin':0, 'Sering':0, 'Lupa_Waktu':0, 'Waktu_Harian':1, 'Usaha_Melepaskan_Diri':0, 'Kesulitan_Melepaskan_Diri':0, 'Butuh_Aplikasi':1, 'Pernah_Memakai_Aplikasi_Pengaturan_Waktu':0
    }, {
        'Jenis_Kelamin':1, 'Sering':2, 'Lupa_Waktu':2, 'Waktu_Harian':1, 'Usaha_Melepaskan_Diri':1, 'Kesulitan_Melepaskan_Diri':1, 'Butuh_Aplikasi':1, 'Pernah_Memakai_Aplikasi_Pengaturan_Waktu':1
    }])
    st.download_button('Download example CSV', data=sample.to_csv(index=False).encode('utf-8'), file_name='fomo_example.csv')
    st.markdown('---')
    st.caption('Made with ❤️ — try different inputs and enjoy exploring!')

# Main Function to switch between different models (for now, FOMO)
if add_selectitem == "FOMO Prediction":
    pass  # we've loaded the UI tabs and logic above
