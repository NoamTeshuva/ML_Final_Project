from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def train(X_train, y_train):
    """
    Trains a Decision Tree model on the provided training data.

    Parameters:
        X_train: Feature matrix for training.
        y_train: Target vector for training.

    Returns:
        A trained Decision Tree classifier pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('decision_tree', DecisionTreeClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == '__main__':
    import numpy as np

    # Create some dummy data for testing:
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = np.random.randint(0, 2, size=100)  # Binary target

    model = train(X_train, y_train)
    print("Decision Tree model trained successfully!")
    print(model)
