def best_loss(y_true, y_pred):
    subtract = y_true - y_pred
    res_square = square(subtract)

def magic_accuracy(y_true, y_pred):
    acc = (((480 * 1) - tf.math.reduce_sum(tf.math.abs(tf.math.subtract(y_pred, y_true)))) / (480 * 1))
