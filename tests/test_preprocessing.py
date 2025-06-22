from __future__ import annotations

import numpy as np
import pytest

from tabpfn.model.preprocessing import ReshapeFeatureDistributionsStep


def test_preprocessing_large_dataset():
    # Generate a synthetic dataset with more than 10,000 samples
    num_samples = 15000
    num_features = 10
    rng = np.random.default_rng()
    X = rng.random((num_samples, num_features))

    # Create an instance of ReshapeFeatureDistributionsStep
    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_norm",
        apply_to_categorical=False,
        append_to_original=False,
        subsample_features=-1,
        global_transformer_name=None,
        random_state=42,
    )

    # Define categorical features (empty in this case)
    categorical_features = []

    # Run the preprocessing step
    result = preprocessing_step.fit_transform(X, categorical_features)

    # Assert the result is not None
    assert result is not None


@pytest.mark.parametrize(
    ("append_to_original_setting", "num_features", "expected_output_features"),
    [
        # Test 'auto' mode below the threshold: should append original features
        pytest.param("auto", 10, 20, id="auto_below_threshold_appends"),
        # Test 'auto' mode above the threshold: should NOT append original features
        pytest.param("auto", 600, 600, id="auto_above_threshold_replaces"),
        # Test True: should always append, regardless of threshold
        pytest.param(True, 600, 1200, id="true_always_appends"),
        # Test False: should never append
        pytest.param(False, 10, 10, id="false_never_appends"),
    ],
)
def test_reshape_step_append_original_logic(
    append_to_original_setting, num_features, expected_output_features
):
    """Tests the `append_to_original` logic, including the "auto" mode which
    depends on the APPEND_TO_ORIGINAL_THRESHOLD class constant (500).
    """
    # ARRANGE: Create a dataset with the specified number of features
    num_samples = 100
    rng = np.random.default_rng(42)
    X = rng.random((num_samples, num_features))

    # ARRANGE: Instantiate the step with the specified setting
    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_norm",
        append_to_original=append_to_original_setting,
        random_state=42,
    )

    # ACT: Run the preprocessing
    Xt, _ = preprocessing_step.fit_transform(X, categorical_features=[])

    # ASSERT: Check if the number of output features matches the expected outcome
    assert Xt.shape[0] == num_samples
    assert Xt.shape[1] == expected_output_features
