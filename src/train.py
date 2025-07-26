import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model():
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.joblib")
    print("✅ Model trained and saved as model.joblib")

if __name__ == "__main__":
    train_model()
