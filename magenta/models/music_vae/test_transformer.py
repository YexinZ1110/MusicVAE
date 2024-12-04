import unittest
import tensorflow as tf
import numpy as np
from music_vae.lstm_models import TransformerDecoder  # Import your TransformerDecoder

class TransformerDecoderTest(unittest.TestCase):
    def setUp(self):
        """Set up the test case with default parameters."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 128
        self.num_heads = 4
        self.num_layers = 2
        self.dff = 256
        self.output_depth = 10  # Number of output classes
        self.max_seq_len = 32
        self.dropout_rate = 0.1
        self.additional_emb = False

        # Create a TransformerDecoder instance
        self.decoder = TransformerDecoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            dropout_rate=self.dropout_rate,
            max_seq_len=self.max_seq_len,
            additional_emb=self.additional_emb
        )

        # Mock hyperparameters and build the decoder
        hparams = tf.contrib.training.HParams(
            batch_size=self.batch_size,
            max_seq_len=self.seq_len
        )
        self.decoder.build(hparams, self.output_depth, is_training=True)

    def test_reconstruction_loss(self):
        """Test the reconstruction_loss method."""
        # Generate mock input and target tensors
        x_input = tf.random.uniform(
            [self.batch_size, self.seq_len, self.output_depth], dtype=tf.float32
        )
        x_target = tf.random.uniform(
            [self.batch_size, self.seq_len, self.output_depth], dtype=tf.float32
        )
        x_length = tf.constant([self.seq_len] * self.batch_size, dtype=tf.int32)

        # Call the reconstruction_loss method
        r_loss, metric_map, logits = self.decoder.reconstruction_loss(
            x_input, x_target, x_length
        )

        # Check output shapes and types
        self.assertEqual(r_loss.shape, (self.batch_size,))
        self.assertTrue(isinstance(metric_map, dict))
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.output_depth))

    def test_sample(self):
        """Test the sample method."""
        n_samples = 3
        max_length = 16

        # Call the sample method
        samples = self.decoder.sample(
            n=n_samples, max_length=max_length, temperature=0.8
        )

        # Check output shapes
        self.assertEqual(samples.shape, (n_samples, max_length, self.d_model))

if __name__ == '__main__':
    unittest.main()
