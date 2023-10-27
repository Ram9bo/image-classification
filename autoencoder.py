import os

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import dataloader


# Define the autoencoder model
def build_autoencoder(input_shape):
    # Encoder
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_layer, decoded)
    return autoencoder, encoded


# Define the paths for the saved models
autoencoder_model_path = 'autoencoder_model.h5'
feature_extractor_model_path = 'feature_extractor_model.h5'

train = dataloader.images()

# Check if the autoencoder model file exists
if os.path.exists(autoencoder_model_path):
    print("Loading the saved autoencoder model...")
    autoencoder = load_model(autoencoder_model_path)
    print("Loading the saved feature extractor...")
    feature_extractor = load_model(feature_extractor_model_path)
else:
    # Create a new autoencoder model if the saved model doesn't exist
    input_shape = (512, 512, 3)
    autoencoder, encoded = build_autoencoder(input_shape)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder on your dataset
    autoencoder.fit(train, train, epochs=10, batch_size=2)

    # Save the entire autoencoder model
    autoencoder.save(autoencoder_model_path)
    print(f"Autoencoder model saved at {autoencoder_model_path}")

    # Create a feature extractor model
    feature_extractor = Model(inputs=autoencoder.input, outputs=encoded)

    # Save the feature extractor model separately
    feature_extractor.save(feature_extractor_model_path)
    print(f"Feature extractor model saved at {feature_extractor_model_path}")

from tensorflow.keras import layers, Sequential

train, val = dataloader.all_data(batch_size=2)

classification_model = Sequential([
    feature_extractor,  # Feature extractor
    layers.Flatten(),  # Flatten the output
    layers.Dense(128, activation='relu'),  # Add dense layers as needed
    layers.Dense(6, activation='softmax')  # Output layer with the number of classes
])

# Compile the classification model

classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classification_model.summary()

classification_model.fit(train, validation_data=val, epochs=15, batch_size=1)

# TODO: try training an autoencoder on all the unlabeled data I have
#   could even try to use the feature extractor as a starting point for pretraining on other AFM data, see if that helps
