import numpy as np
from models.activations import ActivationFunctions

class DenseLayer:
    def __init__(self, input_size, output_size, activation, loss, weight_init='xavier', weight_init_params=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.loss = loss

        if weight_init_params is None:
            weight_init_params = {}

        if weight_init == 'zero':
            self.weights = np.zeros((input_size, output_size))
        elif weight_init == 'uniform':
            lb = weight_init_params.get('lower_bound', -0.1)
            ub = weight_init_params.get('upper_bound', 0.1)
            seed = weight_init_params.get('seed', None)
            
            if seed is not None:
                np.random.seed(seed)
            
            self.weights = np.random.uniform(lb, ub, (input_size, output_size))
        
        elif weight_init == 'normal':
            mean = weight_init_params.get('mean', 0)
            var = weight_init_params.get('variance', 0.1)
            seed = weight_init_params.get('seed', None)
            
            if seed is not None:
                np.random.seed(seed)
            
            self.weights = np.random.normal(
                mean, np.sqrt(var), (input_size, output_size))
        
        elif weight_init == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, 
                limit, (input_size, output_size))
        
        elif weight_init == 'he':
            std = np.sqrt(2 / input_size)
            self.weights = np.random.normal(0, std, (input_size, output_size))
        
        else:
            raise ValueError(
                f"Unsupported weight initialization method: {weight_init}")

        self.biases = np.zeros((1, output_size))
        
        # Initialize gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        
        self.inputs = None
        self.z = None
        self.output = None


    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, dvalues):
        if self.activation.name == 'softmax':
            if self.loss=="categorical_crossentropy":
                dZ = dvalues

            else:
                jacobian_matrices = self.activation.forward(self.z, derivative=True)
                dZ = np.zeros_like(dvalues)
                for i in range(len(dvalues)):
                    dZ[i] = jacobian_matrices[i] @ dvalues[i]
        else:
            dZ = dvalues * self.activation.forward(self.z, derivative=True)
        
        self.dweights = np.dot(self.inputs.T, dZ)
        self.dbiases = np.sum(dZ, axis=0, keepdims=True)
        dinputs = np.dot(dZ, self.weights.T)
        return dinputs

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
