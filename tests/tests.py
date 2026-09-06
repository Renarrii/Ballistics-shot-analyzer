import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from src.main import calculate_angle, draw_skeleton
from src.ml_model import  train_and_evaluate_model


@dataclass
class MockLandmark:
    """Represents an artificial MediaPipe landmark for testing purposes."""
    x: float
    y: float
    z: float
    visibility: float = 1.0


@pytest.fixture
def dummy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Small test-size data for model."""
    X = pd.DataFrame({
        'dip': [120.0, 130.0, 90.0, 85.0, 110.0, 140.0, 88.0, 125.0],
        'release': [45.0, 50.0, 80.0, 85.0, 55.0, 40.0, 82.0, 48.0]
    })
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 0])
    return X, y


# Angle calculation tests

def test_calculate_angle_90_degrees():
    """Checks if the function correctly calculates a right angle."""
    a = MockLandmark(x=0.0, y=1.0, z=0.0)
    b = MockLandmark(x=0.0, y=0.0, z=0.0)
    c = MockLandmark(x=1.0, y=0.0, z=0.0)

    deg, deg_text = calculate_angle(a, b, c)

    assert deg is not None
    assert deg_text is not None
    assert round(deg) == 90
    assert deg_text == "90"


def test_calculate_angle_low_visibility():
    """Checks if the function returns None when the joint visibility is low."""
    a = MockLandmark(x=0.0, y=1.0, z=0.0, visibility=0.9)
    b = MockLandmark(x=0.0, y=0.0, z=0.0, visibility=0.3)
    c = MockLandmark(x=1.0, y=0.0, z=0.0, visibility=0.9)

    deg, deg_text = calculate_angle(a, b, c)

    assert deg is None
    assert deg_text is None


# Drawing tests

def test_draw_skeleton_modifies_frame():
    """Checks if the drawing function actually applies changes to the video frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original_frame = frame.copy()

    draw_skeleton((10, 10), (50, 50), (90, 10), 90.0, "90", frame)

    # Check if the frame is no longer perfectly black
    assert not np.array_equal(frame, original_frame), "The function did not draw anything"


def test_draw_skeleton_ignores_none_values():
    """Checks if the function skips drawing when no angle is provided."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original_frame = frame.copy()

    draw_skeleton((10, 10), (50, 50), (90, 10), None, None, frame)
    assert np.array_equal(frame, original_frame)


# Model tests

def test_train_and_evaluate_model_returns_correct_type(dummy_dataset):
    """Checks if the model trains correctly and can make a prediction."""
    X, y = dummy_dataset

    # Train the model on virtual data
    model = train_and_evaluate_model(X, y)

    # 1. Did the function return the correct object type?
    assert isinstance(model, RandomForestClassifier)

    # 2. Can the model return a result for new, unseen data?
    test_shot = pd.DataFrame([[100.0, 75.0]], columns=['dip', 'release'])
    prediction = model.predict(test_shot)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1], "Prediction must be 0 or 1"
