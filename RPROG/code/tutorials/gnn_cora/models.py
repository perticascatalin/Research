import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def create_ffn(hidden_units, dropout_rate, name = None):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation = tf.nn.gelu))
    return keras.Sequential(fnn_layers, name = name)

def create_baseline_model(hidden_units, num_features, num_classes, dropout_rate = 0.2):
    inputs = layers.Input(shape = (num_features,), name = "input_features")
    x = create_ffn(hidden_units, dropout_rate, name = f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block
        x1 = create_ffn(hidden_units, dropout_rate, name = f"ffn_block{block_idx + 2}")(x)
        # Add skip connection
        x = layers.Add(name = f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits
    logits = layers.Dense(num_classes, name = "logits")(x)
    # Create the model
    return keras.Model(inputs = inputs, outputs = logits, name = "baseline")

def run_experiment(model, x_train, y_train, num_epochs, batch_size, learning_rate):
    # Compile the model
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [keras.metrics.SparseCategoricalAccuracy(name = "acc")])
    # Create an early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor = "val_acc", patience = 50, restore_best_weights = True)
    # Fit the model
    history = model.fit(
        x = x_train,
        y = y_train,
        epochs = num_epochs,
        batch_size = batch_size,
        validation_split = 0.15,
        callbacks = [early_stopping])
    return history