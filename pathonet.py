from tensorflow.keras.models import load_model

model = load_model("pretrained_models/PathoNet.hdf5")

model.summary()
