import socket
import struct
import numpy as np

hostname, sld, tld, port = 'www', 'integralist', 'co.uk', 80
target = '{}.{}.{}'.format(hostname, sld, tld)
d = np.array([0.1, 0.1, 0.1])
# create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect the client
# client.connect((target, port))
client.connect(('0.0.0.0', 9999))

# send some data (in this case a HTTP GET request)
client.send(struct.pack('%sf' % (3), *list(d)))

# receive the response data (4096 is recommended buffer size)

print(struct.unpack('%sf' % (3), client.recv(1024*300)))
