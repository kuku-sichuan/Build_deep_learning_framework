import numpy as np


class Model(object):

    def __init__(self, number_layer, network_config ={}, weight_scale=1e-3, reg=0.0):
        """
        :param network_config: {"input": input_dims(not include batch_size),
                                "1":(layer_type, (c_out, ksize)), ...,"n":}
        :param weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        :param reg: Scalar giving L2 regularization strength
        """
        self.params = {}
        self.reg = reg
        input_dim = network_config["input"]

        C = input_dim
        for i in range(1, number_layer+1):
            if(network_config[str(i)][0] == "conv"):
                num_filters, filter_size = network_config[str(i)][1]
                self.params['W'+str(i)] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
                self.params['b'+str(i)] = np.zeros(num_filters)
                C = num_filters
            elif(network_config[str(i)][0] == "fc"):
                input_size, output_size = network_config[str(i)][1]
                self.params['W' + str(i)] = weight_scale * np.random.randn(input_size, output_size)
                self.params['b' + str(i)] = np.zeros(output_size)

        for k, v in self.params.items():
            self.params[k] = v.astype(np.float32)

    def inference_network(self, X):
        pass

    def backward_network(self):
        pass

    def loss(self, X, y, mode="train"):
        pass
