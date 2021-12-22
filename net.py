import sys
import numpy as np
import time


class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.weights = [np.random.uniform(size=(x, y))-0.5
                        for x, y in zip(layers[1:], layers[:-1])]
        self.biases = [np.random.uniform(size=(x, 1)) for x in layers[1:]]
        # keep a vector of all layers.
        self.vectors = [np.zeros((x)) for x in layers]
        # vectors before activation function.
        self.Zs = [np.zeros((x)) for x in layers]
        self.errors = [np.zeros((x)) for x in layers]
        self.Gs = [np.zeros((x)) for x in layers]
        self.lr = 0.01

    def shuffle_arrays(self, a1, a2):
        """Shuffle two arrays in the same way."""
        rand_state = np.random.get_state()
        np.random.shuffle(a1)
        np.random.set_state(rand_state)
        np.random.shuffle(a2)

    def normalize(self, ndarray):
        """map pixel values in [-0.5,0.5]"""
        return ndarray / 255 - 0.5

    def sigmoid(self, X):
        """The sigmoid function."""
        return 1/(1+np.exp(-X))

    def derSigmoid(self, x):
        """The derivative of the sigmoid function."""
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def softmax(self, X):
        """Softmax function."""
        return np.exp(X)/sum(np.exp(X))

    def predict(self, V):
        """ Predict the given image."""
        self.forward(V.reshape(len(V), 1))
        return np.argmax(self.vectors[-1])

    def test(self, test_x, test_y):
        """Get the accuracy of the test in 0-100%."""
        return 100*np.mean([1 if self.predict(x) == y else 0 for x,
                            y in zip(test_x, test_y)])

    def forward(self, V):
        """for all layers, matrix multiply the weight with the nodes of the
        neural network and add the bias, apply f(*) on the result and save into V."""
        i = 0
        self.vectors[0] = V.copy()
        for W, B in zip(self.weights, self.biases):
            i += 1
            # save all old layers.
            self.Zs[i] = np.dot(W, V)+B
            # On the last layer we want to use softmax and not sigmoid.
            V = self.sigmoid(
                self.Zs[i]) if i < self.num_layers-1 else self.softmax(self.Zs[i])
            self.vectors[i] = V.copy()

    def backward(self, train_x, train_y):
        """Backpropagation process for updating the weights and biases after receiving the loss"""
        for x, y in zip(train_x, train_y):

            x = x.reshape(len(x), 1)
            self.forward(x)

            self.errors[-1] = self.vectors[-1].copy()
            # maybe faster than "one-hot encoding" it.
            self.errors[-1][y] -= 1

            self.Gs[-1] = self.errors[-1].copy()
            # update weights and biases in all layers through chain rule derivatives.
            for l in range(self.num_layers-1, 0, -1):
                delta_w = np.dot(self.Gs[l], self.vectors[l-1].T)
                delta_b = self.Gs[l].copy()
                self.errors[l-1] = np.dot(self.weights[l-1].T, self.Gs[l])
                self.weights[l-1] = self.weights[l-1] - \
                    np.multiply(self.lr, delta_w)
                self.biases[l-1] = self.biases[l-1] - \
                    np.multiply(self.lr, delta_b)
                if l > 1:
                    self.Gs[l -
                            1] = self.derSigmoid(self.Zs[l-1])*self.errors[l-1]

    def train(self, epochs, train_x, train_y):
        """Train the network for epochs times."""
        for i in range(epochs):
            print(f"Epoch #{i+1}", end=' ')
            start = time.process_time()

            self.shuffle_arrays(train_x, train_y)
            self.backward(train_x, train_y)
            end = time.process_time()-start
            print(
                f"finished in {end} seconds or {int((end)/60)} minutes and {int(end%60)} seconds")


def main():

    train_x_path, train_y_path, test_x_path = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(train_x_path)
    train_y = np.loadtxt(train_y_path, dtype=int)
    test_x = np.loadtxt(test_x_path)
    test_y = np.loadtxt(test_x_path, dtype=int)
    print(f"{len(train_x)} data lines loaded")

    # Choose the layer sizes, and how many layers you want.
    layers = [len(train_x[0]), 250, 32,  10]
    neural = NeuralNetwork(layers)

    neural.shuffle_arrays(train_x, train_y)
    print(f"Data has been shuffled.")

    v_percent = 0.1

    train_x_norm = neural.normalize(train_x)
    print(f"Data has been normalized.")
    train_x_norm = train_x_norm[int(v_percent*len(train_x_norm)):]
    train_y = train_y[int(v_percent*len(train_y)):]
    validation_x_normal = train_x_norm[:int(v_percent*len(train_x_norm))]
    validation_y = train_y[:int(v_percent*len(train_y))]
    print(
        f"Data has been seperated to {len(train_x_norm)} train and {len(validation_x_normal)} validation.")

    # Calculate how much time the whole training process took.
    start = time.process_time()
    print("Starting training process...")
    epochs = 15
    neural.train(epochs, train_x_norm, train_y)
    end = time.process_time()-start
    print(
        f"Finished all {epochs} in {end} seconds or {int((end)/60)} minutes and {int(end%60)} seconds")

    # VALIDATION RESULTS
    print(
        f"Validation results: {neural.test(validation_x_normal,validation_y)}%")
    test_x_normalized = neural.normalize(test_x)
    # TEST RESULTS
    print(
        f"Test results: {neural.test(test_x_normalized,test_y)}%")


if __name__ == "__main__":
    main()
