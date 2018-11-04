import numpy as np

"""
L(x_t) = L(x_t-1 + x_s) # x_s is very small
L(x_t) = L(x_t-1) + L'(x_t-1)x_s # Taylor's first order expansion

for L(x_t) < L(x_t-1):
   x_s = - alpha * L'(x_t-1)
"""
def sgd(w, dw, config=None):
    """
    w = w - lr*dw
    :param w: the weight need to update
    :param dw: gradient same size with weight
    :param config: the params of optimor
    :return: the updated weight; and the updated config
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    v = mu * v + (1-mu)dw
    w = w - lr*v
    """

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    mu = config['momentum']
    learning_rate = config['learning_rate']
    v = mu * v + dw
    next_w = w - learning_rate*v

    config['velocity'] = v
    return next_w, config


def rmsprop(x, dx, config=None):
    """
    magnitude = dr*magnitude + (1-dr)*(dx**2)
    w = w - lr*dw/(sqrt(magnitude) + eps)
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    cache = config['cache']
    eps = config['epsilon']
    decay_rate = config['decay_rate']
    learning_rate = config['learning_rate']

    cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
    next_x = x - learning_rate * dx / (np.sqrt(cache) + eps)

    config['cache'] = cache
    return next_x, config


def adam(x, dx, config=None):
    """
    v =  beta1 * v + (1-beta1)*dx
    magnitude = beta2*magnitude + (1-beta2)*(dx**2)
    v = v/(1-beta1**t)
    magnitude = magnitude/(1-beta2**t)
    w = w - lr*(v/sqrt(magnitude+eps))
    """

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.9)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    m = config['m']
    v = config['v']
    t = config['t']
    t += 1
    v = beta1 * v + (1 - beta1) * dx
    m = beta2 * m + (1 - beta2) * (dx ** 2)
    v_bias = v / (1 - beta2 ** t)
    m_bias = m / (1 - beta1 ** t)
    next_x = x - learning_rate * v_bias / (np.sqrt(m_bias) + eps)
    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_x, config