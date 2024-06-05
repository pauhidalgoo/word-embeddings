import tensorflow as tf
from typing import Optional
import numpy as np
from tensorflow.keras import Layer
from tensorflow.keras.regularizers import l2

class MyLayer(Layer):
    def call(self, x):
        return tf.not_equal(x,0)
    
class Cast_layer(Layer):
    def call(self, x):
        return tf.cast(x, tf.float32)
    
class Exp_layer(Layer):
    def call(self, x):
        return tf.exp(x)
    
class Reduce_layer_keep(Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1, keepdims=True)
    
class Reduce_layer(Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

MAX_LEN = 100

def model_1(hidden_size: int = 64,  learning_rate: float = 1e-3) -> tf.keras.Model:
  model = tf.keras.Sequential([
      tf.keras.layers.Concatenate(axis=-1, ),
      tf.keras.layers.Dense(hidden_size, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate))
  return model


def model_2(embedding_size: int = 300, learning_rate: float = 1e-3) -> tf.keras.Model:
    # Capa de entrada para los pares de vectores
    input_1 = tf.keras.Input(shape=(embedding_size,))
    input_2 = tf.keras.Input(shape=(embedding_size,))

    # Hidden layer
    first_projection = tf.keras.layers.Dense(
        embedding_size,
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    projected_1 = first_projection(input_1)
    projected_2 = first_projection(input_2)
    
    # Compute the cosine distance using a Lambda layer
    def cosine_distance(x):
        x1, x2 = x
        x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
        x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
        return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))

    output = tf.keras.layers.Lambda(cosine_distance)([projected_1, projected_2])
    # Define output
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile the model
    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adamax(learning_rate))
    return model

def model_3(embedding_size: int = 300, learning_rate: float = 1e-3) -> tf.keras.Model:
    # Capa de entrada para los pares de vectores
    input_1 = tf.keras.Input(shape=(embedding_size,))
    input_2 = tf.keras.Input(shape=(embedding_size,))

    # Hidden layer
    first_projection = tf.keras.layers.Dense(
        embedding_size,
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    projected_1 =  first_projection(input_1)
    projected_2 = first_projection(input_2)
    
    # Compute the cosine distance using a Lambda layer
    def normalized_product(x):
        x1, x2 = x
        x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
        x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
        return x1_normalized * x2_normalized

    output = tf.keras.layers.Lambda(normalized_product)([projected_1, projected_2])
    output = tf.keras.layers.Dropout(0.1)(output)
    output = tf.keras.layers.Dense(
        16,
        activation="relu",
    )(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(output)
    
    output = tf.keras.layers.Lambda(lambda x: x * 5)(output)
    
    # Define output
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model


def model_4(hidden_size: int = 200, learning_rate: float = 1e-3) -> tf.keras.Model:
  model = tf.keras.Sequential([
      tf.keras.layers.Concatenate(axis=-1, ),
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
      tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.01)),
      tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate))
  return model


def model_5(embedding_size: int = 300, learning_rate: float = 1e-3) -> tf.keras.Model:
    # Input layers for the pairs of vectors
    input_1 = tf.keras.Input(shape=(embedding_size,))
    input_2 = tf.keras.Input(shape=(embedding_size,))

    # Hidden layer with batch normalization and L2 regularization
    first_projection = tf.keras.layers.Dense(
        embedding_size,
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )
    
    projected_1 = first_projection(input_1)
    projected_1 = tf.keras.layers.BatchNormalization()(projected_1)
    projected_2 = first_projection(input_2)
    projected_2 = tf.keras.layers.BatchNormalization()(projected_2)
    
    # Compute the cosine similarity using a Lambda layer
    def cosine_similarity(x):
        x1, x2 = x
        x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
        x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
        return tf.reduce_sum(x1_normalized * x2_normalized, axis=1, keepdims=True)
    
    output = tf.keras.layers.Lambda(cosine_similarity)([projected_1, projected_2])
    
    # Dropout for regularization
    output = tf.keras.layers.Dropout(0.3)(output)
    
    # Fully connected layers with dropout and batch normalization
    output = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)
    
    output = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)
    
    output = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.3)(output)
    
    # Final output layer with scaling
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(output)
    
    output = tf.keras.layers.Lambda(lambda x: x * 5)(output)
    
    # Define and compile the model
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )
    
    return model

def model_6(embedding_size: int = 300, learning_rate: float = 1e-3) -> tf.keras.Model:
    # Input layers for the pairs of vectors
    input_1 = tf.keras.Input(shape=(embedding_size,))
    input_2 = tf.keras.Input(shape=(embedding_size,))

    # Hidden layer with batch normalization and L2 regularization
    first_projection = tf.keras.layers.Dense(
        embedding_size,
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )
    
    projected_1 = first_projection(input_1)
    projected_1 = tf.keras.layers.BatchNormalization()(projected_1)
    projected_2 = first_projection(input_2)
    projected_2 = tf.keras.layers.BatchNormalization()(projected_2)
    
    # Compute the cosine similarity using a Lambda layer
    def cosine_similarity(x):
        x1, x2 = x
        x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
        x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
        return tf.reduce_sum(x1_normalized * x2_normalized, axis=1, keepdims=True)
    
    output = tf.keras.layers.Lambda(cosine_similarity)([projected_1, projected_2])
    
    # Dropout for regularization
    output = tf.keras.layers.Dropout(0.4)(output)
    
    # Fully connected layers with dropout and batch normalization
    output = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.4)(output)
    
    output = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.4)(output)
    
    output = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.4)(output)
    
    # Final output layer with scaling
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(output)
    
    output = tf.keras.layers.Lambda(lambda x: x * 5)(output)
    
    # Define and compile the model
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['mean_squared_error']
    )
    
    return model


def model_7(hidden_size: int = 200, learning_rate: float = 1e-3) -> tf.keras.Model:
    model = tf.keras.Sequential([
    tf.keras.layers.Concatenate(axis=-1),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(1, activation="sigmoid"),
    tf.keras.layers.Lambda(lambda x: x * 5)
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model













#================================================================================================================
#                                           EMBEDING MODELS                                                     #
#================================================================================================================

def model_embeddings_1(
    input_length: int = MAX_LEN,
    dictionary_size: int = 1000,
    embedding_size: int = 100,
    pretrained_weights: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3,
    trainable: bool = False,
    use_cosine: bool = False,
) -> tf.keras.Model:
    # Input layers
    input_1 = tf.keras.Input(shape=(input_length,), dtype=tf.int32)
    input_2 = tf.keras.Input(shape=(input_length,), dtype=tf.int32)

    # Embedding layer
    if pretrained_weights is None:
        embedding = tf.keras.layers.Embedding(
            dictionary_size, embedding_size, input_length=input_length, mask_zero=True
        )
    else:
        dictionary_size = pretrained_weights.shape[0]
        embedding_size = pretrained_weights.shape[1]
        initializer = tf.keras.initializers.Constant(pretrained_weights)
        embedding = tf.keras.layers.Embedding(
            dictionary_size,
            embedding_size,
            input_length=input_length,
            mask_zero=True,
            embeddings_initializer=initializer,
            trainable=trainable,
        )

    # Apply embedding to input sequences
    embedded_1 = embedding(input_1)
    embedded_2 = embedding(input_2)

    # Global average pooling
    _input_mask_1, _input_mask_2 = MyLayer()(input_1), MyLayer()(input_2)
    pooled_1 = tf.keras.layers.GlobalAveragePooling1D()(embedded_1, mask=_input_mask_1)
    pooled_2 = tf.keras.layers.GlobalAveragePooling1D()(embedded_2, mask=_input_mask_2)

    # Compute similarity/distance
    if use_cosine:   
        # Compute the cosine distance using a Lambda layer
        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))
        output = tf.keras.layers.Lambda(cosine_distance)([pooled_1, pooled_2])
    else:
        # Compute the cosine distance using a Lambda layer
        def normalized_product(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return x1_normalized * x2_normalized
    
        output = tf.keras.layers.Lambda(normalized_product)([pooled_1, pooled_2])
        output = tf.keras.layers.Dropout(0.1)(output)
        output = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )(output)
        
        output = tf.keras.layers.Lambda(lambda x: x * 5)(output)

    # Define the model
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate))

    return model


def model_embeddings_2(
    input_length: int = MAX_LEN,
    dictionary_size: int = 1000,
    embedding_size: int = 100,
    learning_rate: float = 1e-3,
    pretrained_weights: Optional[np.ndarray] = None,
    trainable: bool = False,
    use_cosine: bool = False,
) -> tf.keras.Model:
    # Inputs
    input_1 = tf.keras.Input((input_length,), dtype=tf.int32)
    input_2 = tf.keras.Input((input_length,), dtype=tf.int32)

    # Embedding Layer
    if pretrained_weights is None:
        embedding = tf.keras.layers.Embedding(
            dictionary_size, embedding_size, input_length=input_length, mask_zero=True
        )
    else:
        dictionary_size = pretrained_weights.shape[0]
        embedding_size = pretrained_weights.shape[1]
        initializer = tf.keras.initializers.Constant(pretrained_weights)
        embedding = tf.keras.layers.Embedding(
            dictionary_size,
            embedding_size,
            input_length=input_length,
            mask_zero=True,
            embeddings_initializer=initializer,
            trainable=trainable,
        )

    # Embed the inputs
    embedded_1 = embedding(input_1)
    embedded_2 = embedding(input_2)
    # Pass through the embedding layer
    _input_mask_1, _input_mask_2 = input_1, input_2

    # Attention Mechanism
    attention_mlp = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    # Apply attention to each embedding
    attention_weights_1 = attention_mlp(embedded_1)  
    attention_weights_2 = attention_mlp(embedded_2) 
    # Mask the attention weights
    attention_weights_1 = Exp_layer()(attention_weights_1) * Cast_layer()(_input_mask_1[:, :, None])
    attention_weights_2 = Exp_layer()(attention_weights_2) * Cast_layer()(_input_mask_2[:, :, None])
    # Normalize attention weights
    attention_weights_1 = attention_weights_1 / Reduce_layer_keep()(attention_weights_1)
    attention_weights_2 = attention_weights_2 / Reduce_layer_keep()(attention_weights_2)
    # Compute context vectors
    projected_1 = Reduce_layer()(embedded_1 * attention_weights_1) 
    projected_2 = Reduce_layer()(embedded_2 * attention_weights_2) 
    
    
    if use_cosine:
        # Compute the cosine distance using a Lambda layer
        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))
        output = tf.keras.layers.Lambda(cosine_distance)([projected_1, projected_2])
    else:
         # Compute the cosine distance using a Lambda layer
        def normalized_product(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return x1_normalized * x2_normalized
    
        output = tf.keras.layers.Lambda(normalized_product)([projected_1, projected_2])
        output = tf.keras.layers.Dropout(0.1)(output)
        output = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )(output)
        
        output = tf.keras.layers.Lambda(lambda x: x * 5)(output)
    # Model Definition
    model = tf.keras.Model(inputs=(input_1, input_2), outputs=output)
    model.compile(
        loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


def model_embeddings_3(
    input_length: int = MAX_LEN,
    dictionary_size: int = 1000,
    embedding_size: int = 100,
    pretrained_weights: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3,
    trainable: bool = False,
    use_cosine: bool = False,
    l2_reg: float = 1e-4,  # L2 regularization factor
) -> tf.keras.Model:
    # Inputs
    input_1 = tf.keras.Input((input_length,), dtype=tf.int32)
    input_2 = tf.keras.Input((input_length,), dtype=tf.int32)

    # Embedding Layer
    if pretrained_weights is None:
        embedding = tf.keras.layers.Embedding(
            dictionary_size, embedding_size, input_length=input_length, mask_zero=True
        )
    else:
        dictionary_size = pretrained_weights.shape[0]
        embedding_size = pretrained_weights.shape[1]
        initializer = tf.keras.initializers.Constant(pretrained_weights)
        embedding = tf.keras.layers.Embedding(
            dictionary_size,
            embedding_size,
            input_length=input_length,
            mask_zero=True,
            embeddings_initializer=initializer,
            trainable=trainable,
        )

    # Embed the inputs
    embedded_1 = embedding(input_1)
    embedded_2 = embedding(input_2)
    
    # Masking layer to ignore padding tokens
    _input_mask_1, _input_mask_2 = MyLayer()(input_1), MyLayer()(input_2)

    # Attention Mechanism
    attention_mlp = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])

    # Apply attention to each embedding
    attention_weights_1 = attention_mlp(embedded_1)  
    attention_weights_2 = attention_mlp(embedded_2) 
    # Mask the attention weights
    attention_weights_1 = Exp_layer()(attention_weights_1) * Cast_layer()(_input_mask_1[:, :, None])
    attention_weights_2 = Exp_layer()(attention_weights_2) * Cast_layer()(_input_mask_2[:, :, None])
    # Normalize attention weights
    attention_weights_1 = attention_weights_1 / Reduce_layer_keep()(attention_weights_1)
    attention_weights_2 = attention_weights_2 / Reduce_layer_keep()(attention_weights_2)
    # Compute context vectors
    projected_1 = Reduce_layer()(embedded_1 * attention_weights_1) 
    projected_2 = Reduce_layer()(embedded_2 * attention_weights_2) 

    # Dense layers for additional processing with regularization
    dense_layer = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    
    processed_1 = dense_layer(projected_1)
    processed_2 = dense_layer(projected_2)
    
    # Compute similarity/distance
    if use_cosine:
        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))
        output = tf.keras.layers.Lambda(cosine_distance)([processed_1, processed_2])
    else:
        def normalized_product(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return x1_normalized * x2_normalized
    
        output = tf.keras.layers.Lambda(normalized_product)([processed_1, processed_2])
        output = tf.keras.layers.Dropout(0.3)(output)
        output = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Dropout(0.3)(output)
        output = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(output)
        
        output = tf.keras.layers.Lambda(lambda x: x * 5)(output)
        
    # Model Definition
    model = tf.keras.Model(inputs=(input_1, input_2), outputs=output)
    model.compile(
        loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model

def model_embeddings_7(
    input_length: int,
    dictionary_size: int = 1000,
    embedding_size: int = 100,
    pretrained_weights: Optional[np.ndarray] = None,
    learning_rate: float = 1e-3,
    trainable: bool = False,
) -> tf.keras.Model:
    # Input layers
    input_1 = tf.keras.Input(shape=(input_length,), dtype=tf.int32)
    input_2 = tf.keras.Input(shape=(input_length,), dtype=tf.int32)

    # Embedding layer
    if pretrained_weights is None:
        embedding = tf.keras.Embedding(
            input_dim=dictionary_size, output_dim=embedding_size, input_length=input_length, mask_zero=True
        )
    else:
        dictionary_size = pretrained_weights.shape[0]
        embedding_size = pretrained_weights.shape[1]
        initializer = tf.keras.initializers.Constant(pretrained_weights)
        embedding = tf.keras.Embedding(
            input_dim=dictionary_size,
            output_dim=embedding_size,
            input_length=input_length,
            mask_zero=True,
            embeddings_initializer=initializer,
            trainable=trainable,
        )

    # Apply embedding to input sequences
    embedded_1 = embedding(input_1)
    embedded_2 = embedding(input_2)

    # Concatenate the embeddings
    concatenated = tf.keras.Concatenate(axis=-1)([embedded_1, embedded_2])

    # Flatten the concatenated embeddings
    flattened = tf.keras.layers.Flatten()(concatenated)

    # Dense layers
    dense_1 = tf.keras.Dense(2048, activation='relu')(flattened)
    dropout_1 = tf.keras.Dropout(0.5)(dense_1)
    dense_2 = tf.keras.Dense(1024, activation='relu')(dropout_1)
    dropout_2 = tf.keras.Dropout(0.5)(dense_2)
    dense_3 = tf.keras.Dense(256, activation='relu')(dropout_2)
    dropout_3 = tf.keras.Dropout(0.5)(dense_3)
    dense_4 = tf.keras.Dense(64, activation='relu')(dropout_3)
    dropout_4 = tf.keras.Dropout(0.5)(dense_4)
    output = tf.keras.Dense(1)(dropout_4)

    # Define the model
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate))

    return model