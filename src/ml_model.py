import os
import pandas as pd
from typing import Tuple, Any , Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
DATASET_PATH: Optional[str] = os.getenv('DATA_SET_PATH', 'dataset.csv')
MODEL_PATH: Optional[str] = os.getenv('MODEL_PATH', 'model.pkl')
TEST_SIZE: float = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE: int = int(os.getenv('RANDOM_STATE', 42))


def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads the dataset and splits it into features and target."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please run the vision script in GATHER_DATA mode first.")

    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully. Total records: {len(df)}")

    # Checking class balance
    class_counts = df['is_good'].value_counts().to_dict()
    print(f"Class distribution (Good: 1, Bad: 0): {class_counts}")

    required_columns = ['dip', 'release', 'is_good']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in the dataset.")

    X = df[['dip', 'release']]
    y = df['is_good']

    return X, y


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Trains a Random Forest model and evaluates it using comprehensive metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"\nTraining model on {len(X_train)} samples, testing on {len(X_test)} samples...")

    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, class_weight='balanced', random_state=RANDOM_STATE)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy: float = accuracy_score(y_test, predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    print("Confusion Matrix:")

    cm = confusion_matrix(y_test, predictions)
    print(pd.DataFrame(cm, index=['Actually Bad (0)', 'Actually Good (1)'],columns=['Predicted Bad (0)', 'Predicted Good (1)']))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Bad Shot (0)', 'Good Shot (1)']))

    return model


def save_model(model: Any, filepath: str) -> None:
    """Saves the trained model to a specific file path."""
    joblib.dump(model, filepath)
    print(f"\nModel successfully saved to '{filepath}'.")


def main() -> None:
    """Main execution flow for training the basketball model."""
    print("Starting Model Training Process...")

    try:
        X, y = load_and_prepare_data(DATASET_PATH)
        trained_model = train_and_evaluate_model(X, y)
        save_model(trained_model, MODEL_PATH)

    except Exception as error:
        print(f"\n An error occurred during training:\n{error}")


if __name__ == "__main__":
    main()
