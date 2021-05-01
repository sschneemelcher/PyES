import numpy as np
import multiprocessing as mp
import os
import sys
import socket
import threading
import struct
from inspect import isfunction
class ES:
    # ES takes following arguments:
    # - loss:        one of ["mse","acc",custom_function]
    #                the custom function has to take predictions and 
    #                target and must return some kind of number as a score
    # - predict:     a custom function that takes dna and input data and
    #                returns a prediction
    #- predict_args: list of extra arguments that the predict function can take

    def __init__(self, loss, predict, predict_args = [], peers = []):
        self.loss         = loss
        self.predict      = predict
        self.predict_args = predict_args
        self.distributed  = len(peers) > 0
        self.peers        = peers
    
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
            std = np.std(fitness)
            if not std:
                std = 10**-16
        d = (fitness - np.mean(fitness)) / std
        q.put(((lr / (npop * sigma) * np.dot(noise.T, d)), np.amax(fitness)))

    def handle_client_connection(self, client_socket, address):
        self.dna = list(np.array(self.dna) + np.array(struct.unpack('=%sf' % (len(self.dna)), client_socket.recv(8*1024*1300))))
        print("\ngot update from {}:{}".format(address[0], address[1]))
        client_socket.close()

    def listen(self):
        bind_port = 9999
        bind_ip = '0.0.0.0'
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((bind_ip, bind_port))
        server.listen(len(self.peers))
        print("listening on port %d..." %(bind_port))
        while True:
            client_sock, address = server.accept()
            print('Accepted connection from {}:{}'.format(address[0], address[1]))
            client_handler = threading.Thread(
                target=self.handle_client_connection,
                args=(client_sock, address,)
            )
            client_handler.start()


    # - npop:       population size
    # - sigma:      randomness factor for the mutations
    # - lr:         learning rate
    # - epochs:     no of epochs for training
    def fit(self, dna, x_train, y_train, batch_size=0, shuffle=False, lr=0.05, sigma=0.1, npop=50, epochs=10, verbosity=2):
        self.dna = list(dna)
        if self.distributed:
            server = threading.Thread(target=self.listen)
            server.start()
        rows, columns = os.popen('stty size', 'r').read().split()
        cores = mp.cpu_count()
        print("detected %d cores.." % (cores))
        data_len = len(x_train)
        if batch_size < 1:
            batch_size = data_len
        batch_count   = data_len // batch_size
        
        for e in range(epochs):
            if shuffle:
                indices = np.random.permutation(len(x_train))
            else:
                indices = range(data_len)
            
            for b in range(batch_count):
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
                print(d.shape)
                print(dna.shape)
                dna = np.array(self.dna) + d
                self.dna = list(dna)
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                for peer in self.peers:
                    try:
                        client.connect((peer, 9998))
                        client.send(struct.pack('=%sf' % (dna.shape[0]), *list(d)))
                        client.close()
                    except ConnectionRefusedError:
                        print("\npeer %s seems to be down" % (peer))
                if verbosity == 2:
                    if b == batch_count-1:
                        prog = int(np.ceil(b/batch_count)*20)
                    else: 
                        prog = (b*20//batch_count)
                    sys.stdout.write("[%s] epoch %d: batch %d: fitness = %f\n\33[A" % (prog*'#' + (20-prog)*' ', e, b, np.amax(fitness)))
            if verbosity == 1:
                print("epoch %d: fitness = %f" % (e, np.amax(fitness)))
            elif verbosity == 2:
                sys.stdout.write("\n")
        return dna


