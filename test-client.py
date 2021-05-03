import redis
import numpy as np

r = redis.Redis(host='localhost', port=6379, db=0)

dna = np.random.rand(784*64+64*10)
dna_str = str(list(dna))[1:-1]

r.set('hash', hash(dna_str))
r.set('dna', dna_str)

dna2 = np.array(r.get('dna').split(b','), dtype=np.float32)
print(dna2)
