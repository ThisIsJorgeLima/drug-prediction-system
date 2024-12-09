# src/data/data_processor.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        self.le_sex = LabelEncoder()
        self.le_BP = LabelEncoder()
        self.le_chol = LabelEncoder()

    def load_data(self, filepath):
        """Load and preprocess the drug dataset."""
        df = pd.read_csv(filepath)
        X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].copy()
        y = df['Drug']

        # Fit and transform categorical variables
        X['Sex'] = self.le_sex.fit_transform(X['Sex'])
        X['BP'] = self.le_BP.fit_transform(X['BP'])
        X['Cholesterol'] = self.le_chol.fit_transform(X['Cholesterol'])

        return X, y

    def process_input(self, input_data):
        """Process a single input for prediction."""
        processed = input_data.copy()
        processed['Sex'] = self.le_sex.transform([processed['Sex']])[0]
        processed['BP'] = self.le_BP.transform([processed['BP']])[0]
        processed['Cholesterol'] = self.le_chol.transform([processed['Cholesterol']])[0]
        return processed
