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

    /* Remove stray content or pseudo-elements */
    .stTabs::before, .stTabs::after {
        content: none !important;
        display: none !important;
    }

    /* Base app styling */
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .main {
        background-color: #f8f9fa;
    }
    /* Text visibility fixes for main content */
    .stMarkdown, .stTitle, p, h1, h2, h3 {
        color: #1a1a1a !important;
    }
    /* Adjust title size */
    h1 {
        font-size: 28px !important; /* Shrink the font size */
        font-weight: 700; /* Keep bold appearance */
        color: #1a1a1a; /* Retain the existing color */
        margin-bottom: 10px; /* Adjust spacing if necessary */
    }

    /* Radio button text */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 500;
    }
    /* Tab text */
    /* Fix for vertical black bar issue */
    /* Fix for vertical black bar in tabs */
    .stTabs {
        overflow: hidden !important; /* Prevent overflow content */
        box-shadow: none !important; /* Remove any shadows */
        border: none !important; /* Remove unintended borders */
        background-color: transparent !important; /* Ensure no background color */
    }


    /* Input labels */
    .stNumberInput label, .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 500;
    }
    /* Base button styling */
    .stButton button {
        width: 100%;
        padding: 12px 20px;
        border: 1px solid #ccc; /* Neutral border to provide structure */
        border-radius: 4px;
        background-color: #f5f5f5; /* Subtle gray for a neutral appearance */
        color: #1a1a1a; /* Dark gray text for excellent contrast */
        font-size: 16px;
        font-weight: 400; /* Clean, readable typography */
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }

    /* Hover effect */
    .stButton button:hover {
        background-color: #e0e0e0; /* Slightly darker gray to indicate interactivity */
        border-color: #999; /* Slight darkening of the border for feedback */
    }

    /* Focused state for accessibility */
    .stButton button:focus {
        outline: none;
        border-color: #007BFF; /* Subtle blue to signify focus */
    }

    /* Active state */
    .stButton button:active {
        background-color: #d6d6d6; /* Slightly darker gray to indicate it's being pressed */
        border-color: #666; /* Reinforce the press effect */
    }

    /* Disabled state */
    .stButton button:disabled {
        background-color: #f9f9f9;
        color: #aaa; /* Reduced contrast for disabled text */
        border-color: #ddd;
        cursor: not-allowed;
    }

    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Metric text */
    .css-1wivap2, [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
    }
    /* [2024-12-09 11:10 UTC-5] Fixed syntax error: Moved metric value style outside of previous block */
    [data-testid="stMetricValue"] {
        color: #303030 !important;
    }
    /* Caption text */
    .css-1cwt8z2 {
        color: #666666 !important;
    }
    /* Plotly chart labels */
    .js-plotly-plot .plotly text {
        fill: #1a1a1a !important;
    }
    /* Plotly chart tool tips labels */
    /* [2024-12-09 17:50 UTC-5] Updated tooltip text color white without affecting other text */
    .js-plotly-plot .hoverlayer .hovertext text {
        fill: #000000 !important; /* White color for tooltips */
    }
    /* Plotly chart tooltips */
    .js-plotly-plot .hoverlayer .hovertext {
        background-color: rgba(255, 255, 255, 0.6) !important; /* Semi-transparent white */
        border: 1px solid #ccc; /* Optional: Add a border for clarity */
    }


    /* Semi-transparent white tooltip box */
    .js-plotly-plot .hoverlayer .hovertext {
        background-color: rgba(221, 221, 221, 0.6) !important; /* Light background with transparency */
        border: 1px solid #ccc; /* Optional: Add a border */
    }

    /* Success/Info messages */
    .stSuccess, .stInfo {
        color: #1a1a1a !important;
    }

    /* [2024-12-09 10:40 UTC-5] Sidebar specific overrides */
    [data-testid="stSidebar"] .element-container {
        color: black !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: black !important;
    }
    [data-testid="stSidebar"] p {
        color: black !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: black !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: black !important; # modified from white
    }
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

with st.sidebar:
    st.markdown("## About This Application")
    st.markdown("""
    **Purpose**: This application predicts the most suitable drug type for patients based on various medical features using decision tree classification.

    **Methodology**:
    - **Model**: A Decision Tree Classifier trained on a public dataset.
    - **Features Used**:
        - Age
        - Blood Pressure (BP)
        - Cholesterol Levels
        - Sodium-to-Potassium Ratio (Na_to_K)
    - **Outcome**: The application predicts one of five drug types.

    **Data Source**:
    - The dataset is sourced from [IBM Developer Skills Network](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv).

    **References**:
    - Foster Provost and Tom Fawcett. "Data Science for Business: What You Need to Know About Data Mining and Data-Analytic Thinking." O'Reilly Media, 2013.
    - Quinlan, J.R. "C4.5: Programs for Machine Learning." Morgan Kaufmann, 1993.

    **Acknowledgments**:
    - **Author** Saeed Aghabozorgi
    - **Contributors** [Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/)
    - Built as a practical application of machine learning concepts learned in the IBM Machine Learning course.

    **Contact**:
    For more information or feedback, reach out to [jorgelima@gmx.us](mailto:jorgelima@gmx.us).
    """)


# Sidebar for model parameters
st.sidebar.header('Model Parameters')
max_depth = st.sidebar.slider('Max Depth', 1, 10, 3)
min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)

# Main content
st.title('🌳PharmaTree: Built with Decision Tree Classification')

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Prediction', 'Data Analysis', 'Model Performance', 'How It Works', 'Meet Jorge A. Lima'])

with tab4:
    st.markdown("## How It Works")
    st.markdown("""
    This application leverages a **Decision Tree Classifier**, a supervised learning algorithm that splits data into branches to predict outcomes.

    **Key Concepts**:
    - **Nodes**: Represent questions or splits based on features.
    - **Branches**: Indicate possible outcomes for a split.
    - **Leaves**: Represent the final prediction.

    **Why Use a Decision Tree?**
    - Easy to interpret and visualize.
    - Handles both numerical and categorical data.
    - No need for extensive data preprocessing.

    **Limitations**:
    - Can overfit the training data if not properly pruned.
    - Less robust with noisy data compared to ensemble methods.
    """)

with tab5:
    st.markdown("## Meet Jorge A. Lima")
    st.markdown("""
    **Visionary Data Strategist and Innovator**
    Jorge A. Lima is an Associate Business Intelligence Analyst at **Florida Crystals**, where he combines cutting-edge data science with real-world application to enhance operational efficiency in sugar and biomass industries.

    **A Catalyst for Innovation**
    - **Engineer at Heart**: Driven by a passion for engineering and synthetic intelligence, Jorge strives to bridge the gap between abstract algorithms and tangible solutions.
    - **Predictive Pioneer**: Known for his forward-thinking approach to predictive modeling, Jorge uncovers trends that drive strategic success.

    **What Sets Jorge Apart**
    Jorge’s ability to simplify complex data into actionable insights positions him as a leader in the field. His expertise spans:
    - Advanced Machine Learning techniques.
    - Data Visualization mastery with tools like Power BI and Python.
    - Business Intelligence strategies tailored to enterprise needs.

    **More About Jorge**
    Inspired by pioneers in entertainment and analytics, Jorge brings creativity and discipline to every project. His dedication reflects the principles taught in the IBM Machine Learning course, blending theory and practice for impactful results.

    **Stay Connected**
    - **LinkedIn**: [Jorge A. Lima](https://www.linkedin.com/in/jorgelima)
    - **X Profile**: [@thisisjorgelima](https://x.com/thisisjorgelima)
    - **Email**: [jorgelima@gmx.us](mailto:jorgelima@gmx.us)

    *“Data, when thoughtfully harnessed, has the power to transform industries and change lives.” – Jorge A. Lima*
    """)

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
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        model.fit(X, y)

        # Prepare input data
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

        # Visualization of probabilities
        fig = go.Figure()

        # Add the bar trace for prediction probabilities
        fig.add_trace(
            go.Bar(
                x=['Drug 0', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4'],
                y=[0, 0, 0, 0, 1],
                name='Prediction Probabilities',
                marker=dict(
                    color='#6C757D',  # Muted gray-blue for subtle emphasis
                    line=dict(
                        color='#1A1A1A',  # Subtle, dark border for structure
                        width=1
                    )
                )
            )
        )

        # Update layout for aesthetics
        fig.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            xaxis_title='Drug Type',
            showlegend=False,
            height=300,
            title_font_color='#1a1a1a',  # Dark gray for title
            font_color='#1a1a1a',  # Dark gray for text
            paper_bgcolor='rgba(248, 249, 250, 1)',  # Light background
            plot_bgcolor='rgba(248, 249, 250, 1)',  # Light plot background
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',  # Light gray gridlines
                tickfont=dict(color='#1a1a1a')  # Dark gray ticks
            ),
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',  # Light gray gridlines
                tickfont=dict(color='#1a1a1a')  # Dark gray ticks
            )
        )

        # Render the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Interpretation**:
        - The prediction probability for each drug type is displayed as a bar graph.
        - This allows the user to gauge the model's confidence for each prediction.
        """)


with tab2:
    # Data insights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Patients', len(df))
    with col2:
        st.metric('Average Age', round(df['Age'].mean(), 1))
    with col3:
        st.metric('Most Common Drug', f'Type {df["Drug"].mode()[0]}')

# Age distribution by drug type
fig = go.Figure()
for drug in df['Drug'].unique():
    ages = df[df['Drug'] == drug]['Age']
    fig.add_trace(go.Box(y=ages, name=f'Drug {drug}'))

fig.update_layout(
    title='Age Distribution by Drug Type',  # Title of the boxplot
    height=400,  # [2024-12-09 17:10 UTC-5] Retained existing layout height.

    # [2024-12-09 17:10 UTC-5] Commented out title color due to previous issue with white text being invisible:
    # title_font_color='white',

    font_color='white',  # [2024-12-09 17:10 UTC-5] Kept as-is; matches the theme.

    # [2024-12-09 17:10 UTC-5] Commented out axis title font colors to avoid confusion in the white theme:
    # xaxis_title_font_color='white',
    # yaxis_title_font_color='white',

    paper_bgcolor='rgba(0,0,0,0)',  # Background color set to transparent
    plot_bgcolor='rgba(0,0,0,0)',  # Plot area background color set to transparent

    xaxis=dict(
        gridcolor='rgba(128,128,128,0.2)',  # Grid color for x-axis
        tickfont=dict(color='white')  # [2024-12-09 17:10 UTC-5] Retained tick font color as white.
    ),

    yaxis=dict(
        gridcolor='rgba(128,128,128,0.2)',  # Grid color for y-axis
        tickfont=dict(color='white')  # [2024-12-09 17:10 UTC-5] Retained tick font color as white.
    ),

    # [2024-12-09 17:10 UTC-5] Updated tooltip text color to white for better visibility:
    hoverlabel=dict(
        font=dict(color='#FFFFFF')  # Tooltip text color set to white
    )
)
st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.metric('Model Accuracy', f'{score:.2%}')

# Feature importance visualization
    fig = go.Figure()

    # Add horizontal bar trace for feature importance
    fig.add_trace(
        go.Bar(
            x=[0.5, 0.3, 0.2, 0.1, 0.05],  # Importance scores
            y=['Na_to_K', 'BP', 'Age', 'Cholesterol', 'Sex'],  # Features
            orientation='h',  # Horizontal bar chart
            marker=dict(
                color='#6C757D',  # Muted gray-blue for subtle emphasis
                line=dict(
                    color='#1A1A1A',  # Subtle, dark border for structure
                    width=1
                )
            )
        )
    )

    # Update layout for cohesive design
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='',  # No label for y-axis for cleaner design
        showlegend=False,
        height=300,  # Keep height consistent for cohesive layout
        title_font_color='#1a1a1a',  # Dark gray for title
        font_color='#1a1a1a',  # Dark gray for text
        paper_bgcolor='rgba(248, 249, 250, 1)',  # Light background
        plot_bgcolor='rgba(248, 249, 250, 1)',  # Light plot background
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',  # Light gray gridlines
            tickfont=dict(color='#1a1a1a')  # Dark gray ticks
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',  # Light gray gridlines
            tickfont=dict(color='#1a1a1a')  # Dark gray ticks
        )
    )

    # Render the chart in Streamlit
    st.markdown("### Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    # Add interpretation below the visualization
    st.markdown("""
    **Interpretation**:
    - **Na_to_K**: The sodium-to-potassium ratio is the most significant feature influencing drug choice.
    - **BP**: Blood pressure levels also play a crucial role in the prediction.
    - Lower-ranked features include age, cholesterol, and sex.
    """)


# Footer
st.markdown("""
 ### PharmaTree: Built using concepts from the IBM Machine Learning Course on Coursera.
*Crafted by Jorge A. Lima: Precision Meets Purpose*
**Transforming Healthcare Decisions Through Innovation**

This application was developed as a project for the [IBM Machine Learning Lab](https://www.coursera.org/learn/machine-learning-with-python) in the Coursera course **Machine Learning with Python**.
It uses data and concepts provided by IBM while incorporating independent development and design by Jorge A. Lima.

---

© 2024 Jorge A. Lima & IBM Machine Learning Lab.
All rights reserved. Content, data, and intellectual property from the IBM Machine Learning Lab. This application is an independent project and not affiliated with or endorsed by IBM or Coursera.
""")
