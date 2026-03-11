# webapp/app.py

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="PharmaTree: Built with Decision Tree Classification",
    page_icon="🌳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTabs::before, .stTabs::after { content: none !important; display: none !important; }
    .stApp { max-width: 800px; margin: 0 auto; }
    .main { background-color: #f8f9fa; }
    .stMarkdown, .stTitle, p, h1, h2, h3 { color: #1a1a1a !important; }
    h1 { font-size: 28px !important; font-weight: 700; color: #1a1a1a; margin-bottom: 10px; }
    .stRadio label { color: #1a1a1a !important; font-weight: 500; }
    .stTabs { overflow: hidden !important; box-shadow: none !important; border: none !important; background-color: transparent !important; }
    .stNumberInput label, .stSelectbox label { color: #1a1a1a !important; font-weight: 500; }
    .stButton button { width: 100%; padding: 12px 20px; border: 1px solid #ccc; border-radius: 4px; background-color: #f5f5f5; color: #1a1a1a; font-size: 16px; font-weight: 400; cursor: pointer; transition: background-color 0.3s ease, border-color 0.3s ease; }
    .stButton button:hover { background-color: #e0e0e0; border-color: #999; }
    .stButton button:focus { outline: none; border-color: #007BFF; }
    .stButton button:active { background-color: #d6d6d6; border-color: #666; }
    .stButton button:disabled { background-color: #f9f9f9; color: #aaa; border-color: #ddd; cursor: not-allowed; }
    .css-1wivap2, [data-testid="stMetricLabel"] { color: #1a1a1a !important; }
    [data-testid="stMetricValue"] { color: #303030 !important; }
    .css-1cwt8z2 { color: #666666 !important; }
    .js-plotly-plot .plotly text { fill: #1a1a1a !important; }
    .js-plotly-plot .hoverlayer .hovertext text { fill: #000000 !important; }
    .js-plotly-plot .hoverlayer .hovertext { background-color: rgba(221,221,221,0.6) !important; border: 1px solid #ccc; }
    .stSuccess, .stInfo { color: #1a1a1a !important; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('data/drug200.csv')
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['BP'] = le.fit_transform(df['BP'])
    df['Cholesterol'] = le.fit_transform(df['Cholesterol'])
    df['Drug'] = le.fit_transform(df['Drug'])
    X = df.drop('Drug', axis=1)
    y = df['Drug']
    return df, X, y, le


df, X, y, le = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PharmaTree")
    st.markdown("""
    A decision tree classifier predicting drug type from patient attributes.
    Built as part of the IBM Machine Learning course on [Coursera](https://imp.i384100.net/5kgV4b).

    ---

    **Model**: Decision Tree Classifier  
    **Features**: Age, BP, Cholesterol, Na_to_K  
    **Outcome**: One of five drug types  
    **Dataset**: UCI via IBM Developer Skills Network  

    ---

    **References**  
    Provost & Fawcett. *Data Science for Business.* O'Reilly, 2013.  
    Quinlan, J.R. *C4.5: Programs for Machine Learning.* Morgan Kaufmann, 1993.

    **Acknowledgments**  
    Author: Saeed Aghabozorgi  
    Contributor: [Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/)

    ---

    [thisisjorgelima.com](https://thisisjorgelima.com) · [jorgelima@gmx.us](mailto:jorgelima@gmx.us)
    """)
    st.markdown("---")
    st.header('Model Parameters')

max_depth = st.sidebar.slider('Max Depth', 1, 10, 3)
min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title('🌳 PharmaTree: Built with Decision Tree Classification')

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Prediction', 'Data Analysis', 'Model Performance', 'How It Works', 'Jorge A. Lima'])

# ── Tab 1: Prediction ────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        sex = st.radio('Sex', ['F', 'M'])
        bp = st.radio('Blood Pressure', ['LOW', 'NORMAL', 'HIGH'])
        cholesterol = st.radio('Cholesterol', ['NORMAL', 'HIGH'])
    with col2:
        age = st.number_input('Age', 15, 100, 25)
        na_to_k = st.number_input('Sodium to Potassium Ratio', 6.0, 40.0, 15.0)

    if st.button('Predict Drug', type='primary'):
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X, y)
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [1 if sex == 'M' else 0],
            'BP': [{'LOW': 0, 'NORMAL': 1, 'HIGH': 2}[bp]],
            'Cholesterol': [1 if cholesterol == 'HIGH' else 0],
            'Na_to_K': [na_to_k]
        })
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
        st.success(f'Recommended Drug: DrugType {prediction[0]}')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Drug 0', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4'],
            y=proba[0].tolist(),
            marker=dict(color='#6C757D', line=dict(color='#1A1A1A', width=1))
        ))
        fig.update_layout(
            title='Prediction Probabilities', yaxis_title='Probability', xaxis_title='Drug Type',
            showlegend=False, height=300, title_font_color='#1a1a1a', font_color='#1a1a1a',
            paper_bgcolor='rgba(248,249,250,1)', plot_bgcolor='rgba(248,249,250,1)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#1a1a1a')),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#1a1a1a'))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Interpretation**: Bar height shows model confidence for each drug type.")

# ── Tab 2: Data Analysis ─────────────────────────────────────────────────────
with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Patients', len(df))
    with col2:
        st.metric('Average Age', round(df['Age'].mean(), 1))
    with col3:
        st.metric('Most Common Drug', f'Type {df["Drug"].mode()[0]}')

    fig = go.Figure()
    for drug in df['Drug'].unique():
        fig.add_trace(go.Box(y=df[df['Drug'] == drug]['Age'], name=f'Drug {drug}'))
    fig.update_layout(
        title='Age Distribution by Drug Type', height=400, font_color='#1a1a1a',
        paper_bgcolor='rgba(248,249,250,1)', plot_bgcolor='rgba(248,249,250,1)',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#1a1a1a')),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#1a1a1a')),
        hoverlabel=dict(font=dict(color='#1a1a1a'))
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Model Performance ─────────────────────────────────────────────────
with tab3:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    st.metric('Model Accuracy', f'{model.score(X_test, y_test):.2%}')

    importances = model.feature_importances_.tolist()
    features = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    sorted_pairs = sorted(zip(importances, features), reverse=True)
    sorted_importances, sorted_features = zip(*sorted_pairs)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(sorted_importances), y=list(sorted_features), orientation='h',
        marker=dict(color='#6C757D', line=dict(color='#1A1A1A', width=1))
    ))
    fig.update_layout(
        title='Feature Importance', xaxis_title='Importance Score', showlegend=False, height=300,
        title_font_color='#1a1a1a', font_color='#1a1a1a',
        paper_bgcolor='rgba(248,249,250,1)', plot_bgcolor='rgba(248,249,250,1)',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#1a1a1a')),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#1a1a1a'))
    )
    st.markdown("### Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation**:
    - **Na_to_K**: Sodium-to-potassium ratio is the strongest predictor.
    - **BP**: Blood pressure is the second most influential feature.
    - Importances update dynamically based on model parameters.
    """)

# ── Tab 4: How It Works ───────────────────────────────────────────────────────
with tab4:
    st.markdown("## How It Works")
    st.markdown("""
    A **Decision Tree Classifier** — a supervised learning algorithm that splits data into
    branches to predict outcomes based on input features.

    **Key Concepts**  
    - **Nodes** — questions or splits based on features  
    - **Branches** — possible outcomes for each split  
    - **Leaves** — final predictions  

    **Why Decision Tree?**  
    - Easy to interpret and visualize  
    - Handles numerical and categorical data  
    - Minimal preprocessing required  

    **Limitations**  
    - Can overfit without pruning  
    - Less robust with noisy data vs ensemble methods (Random Forest, XGBoost)
    """)

# ── Tab 5: Jorge A. Lima ──────────────────────────────────────────────────────
with tab5:
    st.markdown("## Jorge A. Lima")
    st.markdown("""
    Four countries before age ten. Five generations in sugar. One data platform.

    The work is building real-time data systems for industrial and agricultural operations
    at scale. Currently at **Florida Crystals** — the largest cane sugar producer in the
    United States — architecting the company's first real-time analytics platform across
    194,500 acres of operations.

    Before data: international film production, press coordination at Venice and Cannes,
    diamond grading at HRD Antwerp, cross-border trade across Hong Kong, Shanghai, and
    Beijing. Every role was the same problem — complex information, decisions that can't wait.

    Film taught narrative. Diamonds taught precision. Data engineering is where both converge.

    ---

    **Novus Index** — a self-hosted applied AI research lab. Ubuntu, Docker, Ollama, Qdrant.
    Where the work goes beyond what the day job requires and into what the industry will need next.

    ---

    **Credentials**
    - [IBM AI Engineering Professional Certificate](https://imp.i384100.net/5kgV4b)
    - Data Science & ML — Bloom Institute of Technology
    - Diamond Grader — HRD Antwerp

    [thisisjorgelima.com](https://thisisjorgelima.com) · [LinkedIn](https://www.linkedin.com/in/jorgelima) · [jorgelima@gmx.us](mailto:jorgelima@gmx.us)
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
---
### PharmaTree — IBM Machine Learning Course on [Coursera](https://imp.i384100.net/5kgV4b)
*Crafted by Jorge A. Lima · Precision Meets Purpose*

© 2026 Jorge A. Lima & IBM Machine Learning Lab. Independent project, not affiliated with or endorsed by IBM or Coursera.
""")
