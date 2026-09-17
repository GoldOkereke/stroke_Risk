from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from app.ml.cnn_model import AFibCNN  # noqa: E402


def test_model_architecture_builds():
    model = AFibCNN()
    built = model.build((300, 1))
    assert built.input_shape == (None, 300, 1)
    assert built.output_shape == (None, 1)


def test_prediction_shape_on_dummy_data():
    model = AFibCNN()
    model.build((300, 1))
    X = np.random.normal(0, 1, size=(4, 300, 1)).astype(np.float32)
    probs = model.predict(X)
    assert probs.shape == (4,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_save_and_load_weights_roundtrip(tmp_path):
    model = AFibCNN()
    model.build((300, 1))

    path = tmp_path / "afib_test.keras"
    model.save(path)

    # Our save() appends `.weights.h5` internally if missing.
    expected = tmp_path / "afib_test.weights.h5"
    assert expected.exists()

    model2 = AFibCNN()
    model2.load(path, input_shape=(300, 1))
    probs = model2.predict(np.random.normal(0, 1, size=(2, 300, 1)).astype(np.float32))
    assert probs.shape == (2,)


def test_transfer_learning_freezes_conv_layers(tmp_path):
    base = AFibCNN()
    base.build((300, 1))
    base_path = tmp_path / "ecg.keras"
    base.save(base_path)

    ppg = AFibCNN()
    ppg.load_ecg_weights_for_ppg(base_path, input_shape=(300, 1))

    conv_layers = [layer for layer in ppg.model.layers if layer.__class__.__name__ == "Conv1D"]
    assert conv_layers, "Expected Conv1D layers to exist"
    assert all(layer.trainable is False for layer in conv_layers)
