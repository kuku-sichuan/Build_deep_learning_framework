import numpy as np


def L2_loss(x, y):
    """
    :param x: (N, d1, d2, d3,)
    :param y: same size with x.
    :return a tuple of:
           -loss:
           -dx: same size with x
    """
    x_shape = np.shape(x)
    y = np.reshape(y, x_shape)
    N = np.prod(x_shape)
    loss = np.sum(np.square(x - y)) / N
    dx = np.zeros_like(x)
    dx += 2*(x - y)
    dx /= N

    return loss, dx


def hinge_loss(x, y):
    """
    :param x: (N,k): N is batch size
    :param y: (N,) 0 <=y[i] < K
    :return: -loss
             -dx
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x-correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins>0) / N

    num_pos = np.sum(margins>0, axis=1)
    dx = np.zeros_like(x)
    dx[margins>0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    :param x: (N,k): N is batch size
    :param y: (N,) 0 <=y[i] < K
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x

    Gradient derivation:
    when j != y_i:
     dL/dx = p_i
    when j == y_i:
     dL/dx = p_i -1
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


