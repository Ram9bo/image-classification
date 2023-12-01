import tensorflow as tf


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
