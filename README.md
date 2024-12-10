### README.md

# 🌳 PharmaTree: Data-Driven Insights for Drug Prediction

This repository contains a machine-learning-powered application that predicts the most suitable drug type for patients based on their medical attributes. It leverages a Decision Tree Classifier and was created as part of the IBM Machine Learning Lab, integrated with additional customizations and optimizations.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Dataset](#dataset)
5. [Components](#components)
    - [Application File (`app.py`)](#application-file-apppy)
    - [Drug Predictor (`drug_predictor.py`)](#drug-predictor-drug_predictorpy)
    - [Data Processor (`data_processor.py`)](#data-processor-data_processorpy)
6. [Usage](#usage)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

---

## Overview

The **Drug Decision Assistant** is designed to:
- Predict the best drug for patients based on their medical history and attributes.
- Provide clear visualizations and insights for better interpretability.
- Deliver a user-friendly interface for predictions, data analysis, and model performance.

---

## Features

- **Model:** Decision Tree Classifier.
- **Input Features:**
  - Age
  - Sex
  - Blood Pressure (BP)
  - Cholesterol Levels
  - Sodium-to-Potassium Ratio (Na_to_K)
- **Prediction Outcome:** One of five drug types.
- **Interactive Visualizations:** Displays feature importance and prediction probabilities.
- **Educational Context:** Developed as part of the IBM Machine Learning course on Coursera.

---

## How It Works

The system employs a supervised learning algorithm to classify data and predict the appropriate drug. It:
1. Encodes categorical variables like sex, BP, and cholesterol.
2. Trains a Decision Tree Classifier on labeled data.
3. Predicts the outcome for new patient inputs with accuracy displayed.

---

## Dataset

The dataset used is sourced from the **UCI Machine Learning Repository** and includes:
- Patient attributes such as age, sex, BP, cholesterol, and Na_to_K ratio.
- The target variable, "Drug," categorizes the type of medication.

File: `drug200.csv`

---

## Components

### Application File (`app.py`)

The main Streamlit application file:
- **Handles User Input:** Collects patient information like age, sex, and medical stats.
- **Displays Results:** Predicts drug type and visualizes probabilities.
- **Tabs:** Divides sections into Prediction, Data Analysis, Model Performance, and How It Works.
- **Custom Styling:** Implements a cohesive and user-friendly design with CSS.

---

### Drug Predictor (`drug_predictor.py`)

Defines the machine learning model:
- **Training Function:** Splits the data into training and test sets, trains a Decision Tree Classifier, and returns accuracy.
- **Prediction Function:** Predicts drug type for a given input.

---

### Data Processor (`data_processor.py`)

Handles data preprocessing:
- **Load Data:** Reads and encodes the dataset using LabelEncoder for categorical variables.
- **Process Input:** Prepares user input for prediction by transforming categorical variables.

---

## Usage

### Prerequisites
- Python 3.8 or higher.
- Required libraries: `pandas`, `scikit-learn`, `streamlit`, `plotly`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-url
   ```
2. Navigate to the directory:
   ```bash
   cd drug-decision-assistant
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit application:
```bash
streamlit run webapp/app.py
```

---

## Acknowledgments

- **IBM Machine Learning Lab:** This application was inspired by the IBM course on Coursera. Access the course [here](https://www.coursera.org/learn/machine-learning-with-python).
- **Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

---

## License

This project is a collaborative effort inspired by IBM’s educational resources and data. The dataset and educational context are credited to IBM, while the application code and extensions are the intellectual property of Jorge A. Lima.

© 2024 Jorge A. Lima and IBM Machine Learning Lab.

For inquiries, contact: [thisisjorgelima@gmail.com](mailto:thisisjorgelima@gmail.com) or connect on X: [@thisisjorgelima](https://twitter.com/thisisjorgelima).
