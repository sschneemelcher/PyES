import numpy as np
from PyES import ES
from tensorflow.keras.datasets import mnist

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    r = np.exp(x - np.max(x))
    return r / r.sum()

def predict(dna, x, args):
    pos = 0 
    posb = len(dna)
    for i in range(0, len(args) - 1):
        pos2 = pos + args[i] * args[i+1]
        x = dna[posb-args[i+1]:posb] + np.dot(dna[pos:pos2].reshape(-1, args[i]), x)
        if i != len(args)-1:
            x = relu(x)
        else:
            x = softmax(x)
        pos = pos2
        posb -= args[i+1]
    return np.argmax(x)

if __name__ == '__main__':
    layers = [784, 64, 32, 10]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype("float32")/255).reshape(-1, layers[0])
    x_test = (x_test.astype("float32")/255).reshape(-1, layers[0])
    
    size = 0
    for i in range(len(layers) - 1):
        size += layers[i]*layers[i+1] + layers[i+1]

    dna = np.random.randn(size)

    optimizer = ES("acc", predict, layers)
    dna = optimizer.fit(dna, x_train, y_train, batch_size=500, npop=100, lr=0.1, sigma=0.1, shuffle=True, epochs=10)

    predictions = []
    for x in x_test:
        predictions.append(predict(dna, x, layers))
    print("test accuracy: %f" % (np.mean(predictions == y_test)))
