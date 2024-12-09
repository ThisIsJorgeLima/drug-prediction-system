# webapp/app.py

"""
from src.models.drug_predictor import DrugPredictor
from src.data.data_processor import DataProcessor
import pandas as pd
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    st.set_page_config(
        page_title="Drug Prediction System",
        page_icon="💊",
        layout="wide"
    )

    st.title("Drug Prediction System")
    st.write("Predict suitable drugs based on patient characteristics")

    # Initialize processors
    data_processor = DataProcessor()
    predictor = DrugPredictor()

    # Load and train model
    X, y = data_processor.load_data("data/drug200.csv")
    accuracy = predictor.train(X, y)

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
            sex = st.selectbox("Sex", ["F", "M"])
            bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])

        with col2:
            cholesterol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
            na_to_k = st.number_input(
                "Sodium to Potassium Ratio",
                min_value=0.0,
                max_value=50.0,
                value=15.0
            )

        submitted = st.form_submit_button("Predict Drug")

    if submitted:
        # Process input
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'BP': [bp],
            'Cholesterol': [cholesterol],
            'Na_to_K': [na_to_k]
        })

        processed_input = data_processor.process_input(input_data)
        prediction = predictor.predict(processed_input.values)

        # Display results
        st.success(f"Recommended Drug: {prediction[0]}")
        st.info(f"Model Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
"""
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Drug Prediction System",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        background-color: #2e7d32;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
        /* Text colors */
    .stRadio label, .stTabs label, .stMarkdown, .stTitle {
        color: #000000 !important;
    }
    .stTab > div:first-child {
        color: #000000 !important;
    }
    .stRadio > div {
        color: #000000 !important;
    }
    /* Header text */
    h1, h2, h3 {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and prepare data


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

# Sidebar for model parameters
st.sidebar.header('Model Parameters')
max_depth = st.sidebar.slider('Max Depth', 1, 10, 3)
min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)

# Main content
st.title('Drug Prediction System')

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(['Prediction', 'Data Analysis', 'Model Performance'])

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
        fig = go.Figure(data=[go.Bar(
            x=[f'Drug {i}' for i in range(len(proba[0]))],
            y=proba[0],
            marker_color='rgb(46, 125, 50)'
        )])
        fig.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            xaxis_title='Drug Type',
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

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
        fig.add_trace(go.Violin(y=ages, name=f'Drug {drug}', box_visible=True))
    fig.update_layout(title='Age Distribution by Drug Type', height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.metric('Model Accuracy', f'{score:.2%}')

    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=importance['Importance'],
        y=importance['Feature'],
        orientation='h',
        marker_color='rgb(46, 125, 50)'
    ))
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('---')
st.caption('Drug Prediction System - Powered by Machine Learning')
