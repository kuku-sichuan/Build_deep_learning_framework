import numpy as np


def affine_forward(x, w, b):
    """
    :param x:(N, C1,C2,...Cn)
    :param w: (C1*C2*...*Cn, C_out)
    :param b: (C_out)
    :return: output (N, C_out)
    """
    rex = x.reshape((x.shape[0], -1))
    output = np.dot(rex, w) + b
    cache = (x, w, b)
    return output, cache


def affine_backward(dout, cache):
    """
    :param dout: Upstream derivative, of shape(N, C_out)
    :param cache: Tuple of:
          - x: Input data, of shape(N, C1,C2,...Cn)
          - w: Weights, of shape(C1*C2*...*Cn, C_out)
          - b: bias, of shape(C_out,)
    :return:
    a tuple of:
          - dx: Gradient with respect to x, of shape (N, C1,C2,...Cn)
          - dw: Gradient with respect to w, of shape (C1*C2*...*Cn, C_out)
          - db: Gradient with respect to b, of shape (C_out, )
    """
    x, w, b = cache
    x_shape = x.shape
    rex = x.reshape((x_shape[0], -1))
    '''
    out = x*w + b
    dw = dout*x 
    dx = dout*w 
    db = dout
    '''
    dw = np.dot(rex.T, dout)
    dx = np.dot(dout, w.T)
    db = np.sum(dout, axis=0)
    dx = np.reshape(dx, x_shape)
    return dx, dw, db




