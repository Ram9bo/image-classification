import tensorflow as tf


def obt_accuracy_r(y_true, y_pred):
    """
    Measures the fraction of predictions where the difference between predicted and true regression labels
    is at most 0.1
    """
    threshold = 0.1

    # Calculate the absolute difference between true labels and predictions
    absolute_diff = tf.abs(y_true - y_pred)

    # Create a binary mask indicating if the absolute difference is within the threshold
    within_threshold = tf.cast(absolute_diff <= threshold, tf.float32)

    # Calculate the mean of the binary mask, which gives the percentage within the threshold
    percentage = tf.reduce_mean(within_threshold)

    return percentage


def obh_accuracy_r(y_true, y_pred):
    """
    Measures the fraction of predictions where the difference between predicted and true regression labels
    is at most 0.5
    """
    threshold = 0.5

    # Calculate the absolute difference between true labels and predictions
    absolute_diff = tf.abs(y_true - y_pred)

    # Create a binary mask indicating if the absolute difference is within the threshold
    within_threshold = tf.cast(absolute_diff <= threshold, tf.float32)

    # Calculate the mean of the binary mask, which gives the percentage within the threshold
    percentage = tf.reduce_mean(within_threshold)

    return percentage


def obo_accuracy_r(y_true, y_pred):
    """
    Measures the fraction of predictions where the difference between predicted and true regression labels is at most 1
    """
    threshold = 1.0

    # Calculate the absolute difference between true labels and predictions
    absolute_diff = tf.abs(y_true - y_pred)

    # Create a binary mask indicating if the absolute difference is within the threshold
    within_threshold = tf.cast(absolute_diff <= threshold, tf.float32)

    # Calculate the mean of the binary mask, which gives the percentage within the threshold
    percentage = tf.reduce_mean(within_threshold)

    return percentage


def obo_accuracy(y_true, y_pred):
    """
    Measures the fraction of predictions where the difference between predicted and true classification labels
    is at most 1
    """

    # Calculate the argmax of predicted values to get the predicted class labels
    predicted_labels = tf.argmax(y_pred, axis=-1)

    # Cast y_true to the data type of predicted_labels
    y_true = tf.cast(y_true, predicted_labels.dtype)

    # Calculate the absolute difference between true and predicted class labels
    absolute_difference = tf.abs(y_true - predicted_labels)

    # Check if the absolute difference is less than or equal to 1
    correct_predictions = tf.cast(tf.less_equal(absolute_difference, 1), tf.float32)

    # Calculate the mean accuracy across all predictions
    accuracy = tf.reduce_mean(correct_predictions)

    return accuracy


def accuracy(y_true, y_pred):
    # Calculate the argmax of predicted values to get the predicted class labels
    predicted_labels = tf.argmax(y_pred, axis=-1)

    # Cast y_true to the data type of predicted_labels
    y_true = tf.cast(y_true, predicted_labels.dtype)

    # Check if the absolute difference is less than or equal to 1
    correct_predictions = tf.equal(y_true, predicted_labels)

    # Cast the correct_predictions to float32
    correct_predictions = tf.cast(correct_predictions, tf.float32)

    # Calculate the mean accuracy across all predictions
    accuracy = tf.reduce_mean(correct_predictions)

    return accuracy
