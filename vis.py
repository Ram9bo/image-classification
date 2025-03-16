import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import util

# Plots and saves the loss and accuracy over time, averaged across the provided runs
def save_history_plots(history_df, group, setting):
    # Plot training loss with standard deviation
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='epoch', y='loss', data=history_df, errorbar=('pi', 100), label='Training Loss')
    sns.lineplot(x='epoch', y='val_loss', data=history_df, errorbar=('pi', 100), label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(util.results_path(group, setting, 'plot-loss.png'))

    # Plot training accuracy with standard deviation
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='epoch', y='accuracy', data=history_df, errorbar=('pi', 100), label='Training Accuracy')
    sns.lineplot(x='epoch', y='val_accuracy', data=history_df, errorbar=('pi', 100), label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(util.results_path(group, setting, 'plot-acc.png'))

# Constructs and saves a confusion matrix for the given predictions and labels.
# NOTE: assumes 20 predictions/labels per class
def save_confusion_matrix(all_preds, all_labels, group, setting):
    # Calculate confusion matrices for each run
    cms = [confusion_matrix(labels, preds) for labels, preds in zip(all_labels, all_preds)]

    # Stack confusion matrices and calculate mean and std deviation
    cms = np.array(cms)
    cm_mean = np.round(cms.mean(axis=0) / 20, decimals=2)
    cm_min = np.round(cms.min(axis=0) / 20, decimals=2)
    cm_max = np.round(cms.max(axis=0) / 20, decimals=2)

    print(cm_mean)
    print(cm_max / 20)
    print(cm_min / 20)

    # Create the desired strings using vectorized operations
    result = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(cm_mean.astype(str), '\n ('),
            np.core.defchararray.add(cm_min.astype(str), ', ')
        ),
        np.core.defchararray.add(cm_max.astype(str), ')')
    )

    print(result)

    # Normalize the mean confusion matrix
    cm_mean_normalized = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]
    cm_max_normalized = cm_max.astype('float') / cm_max.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix with standard deviation annotations
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_mean, annot=result, fmt='s', cmap="Blues",
                     xticklabels=[f"Class {i}" for i in range(cm_mean.shape[0])],
                     yticklabels=[f"Class {i}" for i in range(cm_mean.shape[0])])

    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(util.results_path(group, setting, 'normalized_confusion_matrix_with_minmax.png'))
    plt.close()
