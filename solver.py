import numpy as np
import optim


class Solver(object):

    def __init__(self, model, data, **kwargs):

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']

        # Unpack keyword arguments
        # learn and imitate how he do this!the second param is the default value!
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 100)
        self.print_every = kwargs.pop('print_every', 10)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        self.loss_history = []
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        # random choose the samples
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def train(self, num_iterations):
        """
        Run optimization to train the model.
        """
        for t in range(num_iterations):
            self._step()
            # Maybe print training loss
            if (t != 0) and (t % self.print_every == 0):
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))
