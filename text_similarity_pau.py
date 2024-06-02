import tensorflow as tf
def build_and_compile_model(hidden_size: int = 64) -> tf.keras.Model:
  model = tf.keras.Sequential([
      tf.keras.layers.Concatenate(axis=-1, ),
      tf.keras.layers.Dense(hidden_size, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model
m = build_and_compile_model()
# E.g.
import numpy as np
y = m((np.ones((1, 100)), np.ones((1,100)), ), )