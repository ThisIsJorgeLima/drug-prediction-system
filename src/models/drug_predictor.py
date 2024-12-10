# src/models/drug_predictor.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class DrugPredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        self.accuracy = None

    def train(self, X, y):
        """Train the drug prediction model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=3
        )

        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        return self.accuracy

    def predict(self, input_data):
        """Make a prediction for given input."""
        return self.model.predict(input_data.reshape(1, -1))
