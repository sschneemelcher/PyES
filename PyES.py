import numpy as np
import multiprocessing as mp
from inspect import isfunction
class ES:
    # ES takes following arguments:
    # - loss:        one of ["mse","acc",custom_function]
    #                the custom function has to take predictions and 
    #                target and must return some kind of number as a score
    # - predict:     a custom function that takes dna and input data and
    #                returns a prediction
    #- predict_args: list of extra arguments that the predict function can take

    def __init__(self, loss, predict, predict_args = []):
        self.loss         = loss
        self.predict      = predict
        self.predict_args = predict_args
    
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

    def get_fit(self, q, npop, dna, sigma, lr, x_batch, y_batch):
        noise = np.random.randn(npop, len(dna))
        fitness = np.zeros(npop)
        for p in range(npop):
            predictions = []
            model = dna + sigma * noise[p]
            for x in x_batch:
                predictions.append(self.predict(model, x, self.predict_args))
            fitness[p] = self.score(y_batch, predictions)
        d = (fitness - np.mean(fitness)) / np.std(fitness)
        q.put(((lr / (npop * sigma) * np.dot(noise.T, d)), np.amax(fitness)))


    # - npop:       population size
    # - sigma:      randomness factor for the mutations
    # - lr:         learning rate
    # - epochs:     no of epochs for training
    def fit(self, dna, x_train, y_train, batch_size=0, shuffle=False, lr=0.05, sigma=0.1, npop=50, epochs=10, verbosity=2):
        cores = mp.cpu_count()
        print("detected %d cores.." % (cores))
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
                q = mp.Queue()
                procs = []
                for i in range(cores):
                    proc = mp.Process(target=self.get_fit, args=(q, npop//cores, dna, sigma, lr, x_batch, y_batch))
                    procs.append(proc)
                    proc.start()
                fitness, d = 0, 0
                for i in range(len(procs)):
                    content = q.get()
                    d += content[0]
                    fitness = max(fitness, content[1])

                for p in procs:
                    p.join()
                dna += d
                if verbosity == 2:
                    print("epoch %d: batch %d: fitness = %f" % (e, b, np.amax(fitness)))
            if verbosity == 1:
                print("epoch %d: fitness = %f" % (e, np.amax(fitness)))
        return dna


