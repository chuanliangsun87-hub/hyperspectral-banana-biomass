"""
简单的 DNN 回归（使用 Keras）
适用于用谱统计/纹理等 tabular 特征建模
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_dnn(input_dim, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

def train_dnn(model, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)] if X_val is not None else []
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val) if X_val is not None else None,
                        epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=2)
    return model, history
