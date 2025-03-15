import unittest
import numpy as np
from src.models.ffnn import FeedForwardNN
import os


class TestWeights(unittest.TestCase):
    def setUp(self):
        # Inisialisasi model sederhana untuk testing
        self.input_size = 4
        self.hidden_layers = [3]
        self.output_size = 2
        self.activations = ["relu", "softmax"]
        self.model = FeedForwardNN(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            output_size=self.output_size,
            activations=self.activations
        )

    def test_save_load_weights(self):
        # Generate sample input
        X = np.random.randn(5, self.input_size)

        # Get predictions before saving weights
        predictions_before = self.model.forward(X)

        # Save weights
        self.model.save_weights("test_weights.npy")

        # Create new model with same architecture
        new_model = FeedForwardNN(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            output_size=self.output_size,
            activations=self.activations
        )

        # Load weights
        new_model.load_weights("test_weights.npy")

        # Get predictions after loading weights
        predictions_after = new_model.forward(X)

        # Test if predictions are the same
        np.testing.assert_array_almost_equal(
            predictions_before, predictions_after)

        # Cleanup
        if os.path.exists("test_weights.npy"):
            os.remove("test_weights.npy")


if __name__ == '__main__':
    unittest.main()
