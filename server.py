import socket
import threading
import struct
import numpy as np

bind_ip = '0.0.0.0'
bind_port = 9999

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(3)

print('Listening...')

dna = np.array([0.1, 0.4, 5.6])

def handle_client_connection(client_socket):
    global dna
    dna += np.array(struct.unpack('%sf' % (dna.shape[0]), client_socket.recv(1024*300)))
    client_socket.send(struct.pack('%sf' % (dna.shape[0]), *list(dna)))
    client_socket.close()
    

while True:
    client_sock, address = server.accept()
    client_handler = threading.Thread(
        target=handle_client_connection,
        args=(client_sock,) 
    )
    client_handler.start()
