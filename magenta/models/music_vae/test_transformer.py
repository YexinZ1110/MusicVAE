import unittest
import tensorflow as tf
import numpy as np
from magenta.models.music_vae.lstm_models import TransformerDecoder

"""
python -m unittest /Users/shuchenye/Desktop/ESE5460/final-project/MusicVAE/magenta/models/music_vae/test_transformer.py
"""
class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        setattr(self, key, value)


from magenta.models.music_vae.lstm_models import TransformerDecoder

class TransformerDecoderTest(tf.test.TestCase):
    def setUp(self):
        super(TransformerDecoderTest, self).setUp()

        # Define hyperparameters
        self.batch_size = 4
        self.seq_len = 128  # Sequence length for testing
        self.num_layers = 6
        self.d_model = 512
        self.num_heads = 8
        self.dff = 2048
        self.dropout_rate = 0.1
        self.output_depth = 90  # Example output depth (e.g., vocabulary size)

        # Initialize the TransformerDecoder
        self.decoder = TransformerDecoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            dropout_rate=self.dropout_rate,
            max_seq_len=self.seq_len
        )

        # Explicitly call build
        self.decoder.build(hparams=None, output_depth=self.output_depth, is_training=True)


    def test_sample(self):
        max_length = 16  # Max length for sampling
        temperature = 1.0  # Sampling temperature

        # Call sample
        samples = self.decoder.sample(
            n=self.batch_size,
            max_length=max_length,
            temperature=temperature
        )

        # Check the shape of the output
        self.assertEqual(samples.shape[0], self.batch_size)
        self.assertEqual(samples.shape[1], max_length)
        self.assertEqual(samples.shape[2], self.output_depth)  # Ensure output matches expected depth


    def test_reconstruction_loss(self):
        # Create mock data
        x_input = tf.random.uniform(
            [self.batch_size, self.seq_len, self.output_depth], dtype=tf.float32
        )
        x_target = tf.random.uniform(
            [self.batch_size, self.seq_len, self.output_depth], dtype=tf.float32
        )
        x_length = tf.random.uniform(
            [self.batch_size], minval=1, maxval=self.seq_len, dtype=tf.int32
        )

        # Compute reconstruction loss
        r_loss, metric_map, logits = self.decoder.reconstruction_loss(
            x_input, x_target, x_length
        )

        # Validate loss output
        self.assertEqual(r_loss.shape[0], self.batch_size)
        self.assertIn('metrics/reconstruction_loss', metric_map)


if __name__ == '__main__':
    unittest.main()
