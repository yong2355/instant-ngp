import os
import numpy as np
import msgpack


def sparsity(a):
 mip = a.shape[0]
 size = a.shape[1] ** 3
 for i in range(mip):
  w = a[i]
  w = w.reshape(size)
  s = 100. * float(np.sum(w == False)) / float(size)
  print(f"[*] Mip{i} sparsity = {s}%")
#### msgpack handler ####
# msgpack to dictionary
def load_msgpack(path):
 cwd = os.getcwd()
 fpath = os.path.join(cwd, path)
 with open(fpath, 'rb') as data_file:
  packed_config = data_file.read()

 config = msgpack.unpackb(packed_config)

 return config 

# dictionary to msgpack
def save_msgpack(path, config):
 packed_config = msgpack.packb(config)

 cwd = os.getcwd()
 fpath = os.path.join(cwd, path)
 with open(fpath, "wb") as out_file:
  out_file.write(packed_config)

#### helper functions ####
def next_multiple(val, divisor):
 tmp = int((val + divisor - 1) / divisor)
 result = tmp * divisor

 return result

