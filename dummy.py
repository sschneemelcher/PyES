import numpy as np
from PyES import ES
from tensorflow.keras.datasets import mnist

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    r = np.exp(x - np.max(x))
    return r / r.sum()

def predict(dna, x, args):
    x = relu(np.dot(dna[:args[0]*args[1]].reshape(-1, args[0]), x))
    for i in range(1, len(args) - 1):
        x = relu(np.dot(dna[args[i-1] * args[i]:args[i-1] * args[i] + args[i] * args[i+1]].reshape(-1, args[i]), x))
    return np.argmax(softmax(np.dot(x, dna[-(args[-2] * args[-1]):].reshape(args[-1], -1))))

if __name__ == '__main__':
    layers = [784, 64, 32, 10]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype("float32")/255).reshape(-1, layers[0])
    x_test = (x_test.astype("float32")/255).reshape(-1, layers[0])
    
    size = 0
    for i in range(len(layers) - 1):
        size += layers[i]*layers[i+1]

    dna = np.random.randn(size)

    optimizer = ES("acc", predict, layers)
    dna = optimizer.fit(dna, x_train, y_train, batch_size=2000, npop=100, shuffle=True, epochs=10)

    predictions = []
    for x in x_test:
        predictions.append(predict(dna, x, layers))
    print("test accuracy: %f" % (np.mean(predictions == y_test)))
