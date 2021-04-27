import numpy as np
from inspect import isfunction
class ES:
    # ES takes following arguments:
    # - loss:       one of ["mse","acc",custom_function]
    #               the custom function has to take predictions and 
    #               target and must return some kind of number as a score
    # - predict:    a custom function that takes dna and input data and
    #               returns a prediction
    def __init__(self, loss, predict):
        self.loss       = loss
        self.predict    = predict

    def mse(self, y_true, y_pred):
        return -np.mean((y_true - y_pred)**2)

    def accuracy(self, y_true, y_pred):
        return np.mean((y_true == y_pred))

    def score(self, y_true, y_pred):
        if isfunction(self.loss):
            return self.loss(y_true, y_pred)
        elif self.loss == "mse":
            return self.mse(y_true, y_pred)
        elif self.loss == "acc":
            return self.accuracy(y_true, y_pred)
        else:
            raise ValueError('Unknown loss')

    # - npop:       population size
    # - sigma:      randomness factor for the mutations
    # - lr:         learning rate
    # - epochs:     no of epochs for training
    def fit(self, dna, x_train, y_train, batch_size=0, shuffle=False, lr=0.05, sigma=0.1, npop=50, epochs=10, verbosity=2):
        data_len = len(x_train)
        if batch_size < 1:
            batch_size = data_len

        for e in range(epochs):

            if shuffle:
                indices = np.random.permutation(len(x_train))
            else:
                indices = range(data_len)

            for b in range(data_len//batch_size):
                x_batch = x_train[indices[b*batch_size:(b+1)*batch_size]]
                y_batch = y_train[indices[b*batch_size:(b+1)*batch_size]]
                fitness = np.zeros(npop)
                noise = np.random.randn(npop, len(dna))

                for n in range(npop):
                    predictions = []
                    model = dna + sigma * noise[n]
                    for x in x_train:
                        predictions.append(self.predict(model, x))
                    fitness[n] = self.score(y_train, predictions)
                d = (fitness - np.mean(fitness)) / np.std(fitness)
                dna = dna + lr / (npop * sigma) * np.dot(noise.T, d)
                
                if verbosity == 2:
                    print("epoch %d: batch %d: fitness = %f" % (e, b, np.amax(fitness)))
            if verbosity == 1:
                print("epoch %d: fitness = %f" % (e, np.amax(fitness)))
        return dna
