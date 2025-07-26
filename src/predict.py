import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def make_prediction():
    # Load model
    model = joblib.load("model.joblib")

    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Take a small sample for prediction
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict first 5 samples
    predictions = model.predict(X_test[:5])
    print("✅ Predictions for first 5 samples:", predictions)

if __name__ == "__main__":
    make_prediction()
