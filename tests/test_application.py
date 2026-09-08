import numpy as np
import pytest

from application import save_model, load_model, preprocess_image, split_training
from network import NeuralNetwork


def test_checkpoint_preserves_batchnorm_and_predictions(tmp_path):
    np.random.seed(7)
    model = NeuralNetwork(
        hidden_sizes=[8], use_batchnorm=True, use_dropout=True, dropout_ratio=0.2
    )
    x = np.random.random((6, 784))
    model.forward(x, train=True)
    before = model.predict(x).copy()
    path = tmp_path / "model.npz"
    save_model(model, path, {"seed": 7})
    restored, metadata = load_model(path)
    np.testing.assert_array_equal(restored.predict(x), before)
    assert metadata["seed"] == 7


def test_split_is_fixed_disjoint_and_has_no_test_dependency():
    a, b = split_training(60000, seed=42)
    c, d = split_training(60000, seed=42)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert len(a) == 50000 and len(b) == 10000
    assert not set(a) & set(b) and len(set(a) | set(b)) == 60000


def test_blank_and_polarity_preprocessing():
    from PIL import Image, ImageDraw

    with pytest.raises(ValueError, match="blank"):
        preprocess_image(Image.new("L", (100, 100), 0))
    with pytest.raises(ValueError, match="blank"):
        preprocess_image(Image.new("L", (100, 100), 255))
    im = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(im).line((50, 10, 50, 90), fill=255, width=10)
    pixels = preprocess_image(im)
    assert pixels.shape == (1, 784) and pixels.min() >= 0 and pixels.max() <= 1
    np.testing.assert_array_equal(
        pixels, preprocess_image(Image.fromarray(255 - np.asarray(im)))
    )


def test_full_mlp_gradient_without_dropout_matches_finite_difference():
    np.random.seed(9)
    model = NeuralNetwork(hidden_sizes=[4], use_batchnorm=False, use_dropout=False)
    x = np.random.random((3, 784))
    y = np.array([1, 3, 5])
    model.gradient(x, y)
    for key, index in [("W1", (9, 1)), ("b1", (1,)), ("W2", (1, 3)), ("b2", (3,))]:
        analytical = float(model.grads[key][index])
        old = model.params[key][index]
        eps = 1e-5
        model.params[key][index] = old + eps
        plus = model.loss(x, y)
        model.params[key][index] = old - eps
        minus = model.loss(x, y)
        model.params[key][index] = old
        assert analytical == pytest.approx((plus - minus) / (2 * eps), abs=2e-6)
