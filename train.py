"""
Model training.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# from tensorflow.keras.applications.resnet50 import preprocess_input
import dataloader
import network

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def train_network():
    # model = network.compile_model()
    # model = network.resnet()
    # model = network.vgg16()
    model = network.xception()

    (cifar_train_x, cifar_train_y), (cifar_test_x, cifar_test_y) = dataloader.cifar_data()
    train, val = dataloader.all_data()
    # train = preprocess_input(train)
    # TODO: this preprocessing may be useful for the pretrained resnet but apparently can't be done on the
    #  dataset object so might be something to do in the dataloader

    # TODO: manually pretrain on a dataset other than imagenet (ideally the same sort of microscopy could also be in
    #  combination with imagenet)

    # history = model.fit(train_data, epochs=20, verbose=1, validation_data=test_data)
    # model.fit(x=cifar_train_x, y=cifar_train_y, epochs=10, verbose=1, validation_data=(cifar_test_x, cifar_test_y))
    return model.fit(train, epochs=10, verbose=1, validation_data=val)


# TODO: create averaged plots for cifar with different subset sizes of the training data (full test set can be used)
#  to illustrate accuracy gains on larger datasets and make estimates regarding the amount of extra data needed

# TODO: set up a BO-HPO experiment to optimize the architecture and hyperparameters

def run_cifar():
    model = network.resnet(input_shape=(32, 32, 3), num_classes=10)
    (cifar_train_x, cifar_train_y), (cifar_test_x, cifar_test_y) = dataloader.cifar_data()
    sublength = 5000
    cifar_train_x = cifar_train_x[:sublength]
    cifar_train_y = cifar_train_y[:sublength]

    return model.fit(x=cifar_train_x, y=cifar_train_y, epochs=30, verbose=1,
                     validation_data=(cifar_test_x, cifar_test_y))


def test_and_train():
    # model = network.compile_model()
    model = network.resnet()
    train, val = dataloader.all_data()

    # history = model.fit(train_data, epochs=20, verbose=1, validation_data=test_data)
    # model.fit(x=cifar_train_x, y=cifar_train_y, epochs=10, verbose=1, validation_data=(cifar_test_x, cifar_test_y))
    model.fit(train, epochs=30, verbose=1, validation_data=val)
    preds = np.argmax(model.predict(val), axis=1)
    print(preds)
    labels = []
    for i, l in val:
        labels += list(l.numpy())
    labels = np.array(labels)
    print(labels)
    conf_mat = np.round(tf.math.confusion_matrix(labels, preds) / len(preds), decimals=2)
    print(conf_mat)


def average_train():
    # Assuming you have a 'history' object returned by model.fit()
    # Replace 'history' with the actual name of your history object
    # Create a Seaborn DataFrame from the history
    histories = [pd.DataFrame(train_network().history) for i in range(3)]
    history_df = pd.concat(histories)
    print(history_df)
    history_df.to_csv("efficientb0.csv", index=False)
    # Plot the loss
    colors = ['blue', 'red', 'green']
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=history_df[['loss', 'val_loss']], palette=colors)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    # Save the loss plot as an image
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Plot accuracy (if available in the history)
    if 'accuracy' in history_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=history_df[['accuracy', 'val_accuracy']], palette=colors)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Training Accuracy', 'Validation Accuracy'])

        # Save the accuracy plot as an image
        plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
        plt.close()


average_train()
# test_and_train()
# run_cifar()
# TODO: do a 5-run average (50-50 data split) for: no data augment, only rotation, only flipping, full augmentation,
#  in combination with CNN, ResNet


# TODO: try ensemble models
# TODO: once we have more data we can also train those models on separate parts of the data
# TODO: maybe it could even work on differently augmented data sets
