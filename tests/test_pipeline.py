
import joblib
import numpy as np
from sklearn.datasets import load_wine

from src.clustering import build_wine_pipeline


def test_saved_pipeline_predictions_are_identical(tmp_path):
    # Load the Wine dataset
    X, _ = load_wine(return_X_y=True)

    # Build and train the pipeline
    pipeline = build_wine_pipeline(n_clusters=3, variance=0.95)
    pipeline.fit(X)

    # Get predictions before saving
    predictions_before = pipeline.predict(X)

    # Save the trained pipeline
    model_path = tmp_path / "wine_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    # Reload the saved pipeline
    loaded_pipeline = joblib.load(model_path)

    # Get predictions after reloading
    predictions_after = loaded_pipeline.predict(X)

    # Confirm both sets of predictions are identical
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after
    )
