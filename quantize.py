import math
import matplotlib.pyplot as plt
import scipy.stats as sp
import numpy as np

import utils

#### initialize model ####
# msgpack dictionary to model dictionary
def init_model(mode, config, params):
  #print(f"[*] initializing model...")
 model = dict()
 offset = 0
 if mode == 'nerf':
  print("*****density network")
  model['density_network'], newparams1 = init_mlp(mode, config['network'], params, offset)
  offset += model['density_network']['n_params']
  print("WonRaeItDon Print")
  print(model['density_network']['n_params'])
  print("*****rgb network")
  model['rgb_network'], newparams2 = init_mlp(mode, config['rgb_network'], newparams1, offset)
  offset += model['rgb_network']['n_params']
  print("WonRaeItDon Print")
  print(model['rgb_network']['n_params'])
  print("*****pos encoding")
  model['pos_encoding'], newparams3, scales = init_encoding(mode, config['encoding'], newparams2, offset)
  offset += model['pos_encoding']['n_params']
  print("WonRaeItDon Print")
  print(model['pos_encoding']['n_params'])
 else:
  model['network'] = init_mlp(mode, config['network'], params, offset)
  offset += model['network']['n_params']
  model['encoding'], newparams, scales = init_encoding(mode, config['encoding'], params, offset)
  offset += model['encoding']['n_params']
 return model, newparams3, scales

def init_mlp(mode, network_config, params, pointer):
 network = dict()
 #1. init hyperparams from the config
 # FIXME: input_width/output_width are set implicitly
 input_width = int() 
 output_width = int()
 newparams = params.copy()
 if mode == 'nerf':
  input_width = 32
  # FIXME: distinguish between density network & rgb network
  if network_config['n_hidden_layers'] == 1: 
   output_width = 16 # density network
  else:
   output_width = 3 # rgb network
 elif mode == 'sdf':
  input_width = 32 # L * F
  output_width = 1
 elif mode == 'image':
  input_width = 32
  output_width = 3
 else:
  raise ValueError("Must specify a valid '--mode'")
 tensorcore_width = 16 # FIXME: tensorcore_width depends on GPU arch (e.g. SM75: 8)
 padded_output_width = utils.next_multiple(output_width, tensorcore_width)
 hidden_width = network_config['n_neurons']
 activation = network_config['activation']
 output_activation = network_config['output_activation']
 n_hidden_layers = network_config['n_hidden_layers']
 # 2. generate model dictionary
 network['input_width'] = input_width 
 #print("input_width")
 #print(input_width)
 network['hidden_width'] = hidden_width
 #print("hidden_width")
 #print(hidden_width)
 network['output_width'] = output_width
 #print("output_width")
 #print(output_width)
 network['padded_output_width'] = padded_output_width
 #print("padded_output_width")
 #print(padded_output_width)
 network['activation'] = activation    
 #print("activation")
 #print(activation)
 network['output_activation'] = output_activation    
 #print("output_activation")
 #print(output_activation)
 network['n_hidden_layers'] = n_hidden_layers
 scales=[]
 #print("n_hidden_layers")
 #print(n_hidden_layers)
 # FIXME: for now weights are stored as a list of numpy arrays
 network_params = [] 
 n_hidden_matmuls = 0
 q_range = 255
 max_int = 127
 min_int = -128
 if n_hidden_layers > 0:
  n_hidden_matmuls = n_hidden_layers - 1
 original_pointer = pointer # FIXME
 if n_hidden_layers == 0:
  # output layer
  print("**output layer")
  #print("n_params_in_lev")
  n_params_in_lev = padded_output_width * input_width 
  #print(n_params_in_lev)
  network_params.append(params[pointer:pointer+n_params_in_lev])
  n, bins, _ = plt.hist(params[pointer:pointer+n_params_in_lev])
  print("n, bins")
  print(n)
  print(bins)
  print("min, max")
  print(min(params[pointer:pointer+n_params_in_lev]))
  print(max(params[pointer:pointer+n_params_in_lev]))
  pointer += n_params_in_lev
 else:
  # input layer
  print("**input layer")
  #print("n_params_in_lev")
  n_params_in_lev = hidden_width * input_width 
  #print(n_params_in_lev)
  network_params.append(params[pointer:pointer+n_params_in_lev])
  n, bins, _ = plt.hist(params[pointer:pointer+n_params_in_lev])
  print("n, bins")
  print(n)
  print(bins)
  print("min, max")
  print(min(params[pointer:pointer+n_params_in_lev]))
  print(max(params[pointer:pointer+n_params_in_lev]))
  scale_input = np.float16(2*max(params[pointer:pointer+n_params_in_lev])/q_range) if max(params[pointer:pointer+n_params_in_lev]) > -min(params[pointer:pointer+n_params_in_lev]) else np.float16(-2*min(params[pointer:pointer+n_params_in_lev])/q_range)
  quan_input = np.clip(np.round(newparams[pointer:pointer+n_params_in_lev]/scale_input),-128,127)
  newparams[pointer:pointer+n_params_in_lev]=quan_input
  print("SSSSS")
  print(newparams[pointer:pointer+n_params_in_lev])
  #newparams[pointer:pointer+n_params_in_lev]=quan_input*scale_input
  scales.append(scale_input)
  pointer += n_params_in_lev
                                                                                                                                                                                                                                                                                                                                                                                                                                                           
  # hidden layers
  for i in range(n_hidden_matmuls):
   print("**hidden layer")
   print("n_params_in_lev")
   n_params_in_lev = hidden_width * hidden_width 
   print(n_params_in_lev)
   network_params.append(params[pointer:pointer+n_params_in_lev])
   n, bins, _ = plt.hist(params[pointer:pointer+n_params_in_lev])
   print("n, bins")
   print(n)
   print(bins)
   print("min, max")
   print(min(params[pointer:pointer+n_params_in_lev]))
   print(max(params[pointer:pointer+n_params_in_lev]))
   scale_hidden = np.float16(2*max(params[pointer:pointer+n_params_in_lev])/q_range) if max(params[pointer:pointer+n_params_in_lev]) > -min(params[pointer:pointer+n_params_in_lev]) else np.float16(-2*min(params[pointer:pointer+n_params_in_lev])/q_range)
   quan_hidden = np.clip(np.round(newparams[pointer:pointer+n_params_in_lev]/scale_hidden),-128,127)
   newparams[pointer:pointer+n_params_in_lev]=quan_hidden
   #newparams[pointer:pointer+n_params_in_lev]=quan_hidden*scale_hidden
   print("SSSSS")
   print(newparams[pointer:pointer+n_params_in_lev])
   scales.append(scale_hidden)
   pointer += n_params_in_lev
 
  # output layer
  print("**output layer")
  #print("n_params_in_lev")
  n_params_in_lev = padded_output_width * hidden_width 
  #print(n_params_in_lev)
  network_params.append(params[pointer:pointer+n_params_in_lev])
  n, bins, _ = plt.hist(params[pointer:pointer+n_params_in_lev])
  print("n, bins")
  print(n)
  print(bins)
  print("min, max")
  print(min(params[pointer:pointer+n_params_in_lev]))
  print(max(params[pointer:pointer+n_params_in_lev]))
  scale_output = np.float16(2*max(params[pointer:pointer+n_params_in_lev])/q_range) if max(params[pointer:pointer+n_params_in_lev]) > -min(params[pointer:pointer+n_params_in_lev]) else np.float16(-2*min(params[pointer:pointer+n_params_in_lev])/q_range)
  quan_output = np.clip(np.round(newparams[pointer:pointer+n_params_in_lev]/scale_output),-128,127)
  newparams[pointer:pointer+n_params_in_lev]=quan_output
  #newparams[pointer:pointer+n_params_in_lev]=quan_output*scale_output
  print("SSSSS")
  print(newparams[pointer:pointer+n_params_in_lev])
  scales.append(scale_output)
  pointer += n_params_in_lev

 network['params'] = network_params
 network['offset'] = original_pointer
 network['n_params'] = pointer - original_pointer # FIXME 

 print("scale factors!!!!!!!!!!")
 np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.50f}".format(x)})
 print(np.float16(scales))

 return network, newparams

def init_encoding(mode, encoding_config, params, pointer):
 # 1. initialize hyperparameters from the config
 #print("n_features_per_level")
 F = encoding_config['n_features_per_level']
 #print(F)
 #print("n_levels")
 L = encoding_config['n_levels']
 #print(L)
 #print("T_max")
 T_max = 1 << encoding_config['log2_hashmap_size']
 #print(T_max)
 #print("base_resolution")
 N_min = encoding_config['base_resolution']
 #print(N_min)
 # FIXME: hyperparams that depend on the workload
 DIM = int() 
 N_max = int() 
 # FIXME: change aabb_scale & max_res
 aabb_scale = 4 # for fox
 max_res = 1024 # for albert
 if mode == 'nerf':
  DIM = 3
  #N_max = 2048 * encoding_config['aabb_scale'] # output of density network
  #print("N_max")
  N_max = 2048 * aabb_scale # output of density network
  #print(N_max)
 elif mode == 'sdf':
  DIM = 3
  N_max = 2048
 elif mode == 'image':
  DIM  = 2
  N_max = max_res / 2
 else:
  raise ValueError("Must specify a valid '--mode'")
 b = math.exp(math.log(N_max / N_min) / (L-1))  
 HASHMAP_OFFSET_TABLE = []
 offset = 0
 for lev in range(L):
  scale = N_min * (b ** lev) - 1.0
  # FIXME: due to fp computation error
  N_lev = int()
  tmp = int(math.ceil(scale))
  if (tmp & (tmp - 1)): 
   N_lev = tmp + 1
  else: 
   N_lev = tmp
  # FIXME: watch out for the overflow
  T_lev = utils.next_multiple(N_lev ** DIM, 8)
  T_lev = min(T_lev, T_max)
  HASHMAP_OFFSET_TABLE.append(offset)
  offset += T_lev
 HASHMAP_OFFSET_TABLE.append(offset)
 # 2. generate model dictionary
 encoding = dict()
 encoding['F'] = F
 encoding['L'] = L
 encoding['T_max'] = T_max
 encoding['N_min'] = N_min
 encoding['N_max'] = N_max
 encoding['b'] = b
 encoding['dim'] = DIM
 #print("Hashmap_offset_table")
 #print(HASHMAP_OFFSET_TABLE)
 encoding['hashmap_offset_table'] = HASHMAP_OFFSET_TABLE
 #print("params")
 #print(encoding['params'])
 #print("n_params")
 encoding['n_params'] = HASHMAP_OFFSET_TABLE[-1]*F
 scales = []
 
 q_range = 255
 q_minimum = -128
 q_maximum = 127
 print("level 1")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F])/q_range)
 print("level 2")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F])/q_range)
 print("level 3")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F])/q_range)
 print("level 4")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F])/q_range)
 print("level 5")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F])/q_range)
 print("level 6")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F])/q_range)
 print("level 7")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F])/q_range)
 print("level 8")
 n, bins, _ = plt.hist(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F])
 print("n, bins")
 print(n)
 print(bins)
 print(min(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F]), max(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F]))
 scales.append(2*max(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F])/q_range if max(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F])>-min(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F]) else -2*min(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F])/q_range)
 #################################################asymetric quantization to INT8
 scale_lev1 = np.float16(scales[0])
 print("scale typeeeeeeeeeeeeeeeeeeee")
 #print(format(np.float32(scales[0]),"b"))
 print(np.float16(scales))
 np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.50f}".format(x)})
 print(np.float16(scales))
 #np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.50f}".format(x)})
 print(scales)
 bias_lev1 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F])/scale_lev1)+q_minimum
 scale_lev2 = np.float16(scales[1])
 bias_lev2 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F])/scale_lev2)+q_minimum
 scale_lev3 = np.float16(scales[2])
 bias_lev3 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F])/scale_lev3)+q_minimum
 scale_lev4 = np.float16(scales[3])
 bias_lev4 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F])/scale_lev4)+q_minimum
 scale_lev5 = np.float16(scales[4])
 bias_lev5 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F])/scale_lev5)+q_minimum
 scale_lev6 = np.float16(scales[5])
 bias_lev6 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F])/scale_lev6)+q_minimum
 scale_lev7 = np.float16(scales[6])
 bias_lev7 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F])/scale_lev7)+q_minimum
 scale_lev8 = np.float16(scales[7])
 bias_lev8 = -round(min(params[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F])/scale_lev8)+q_minimum
 newparams = params.copy()
 print(type(newparams[pointer]))
 print("scale and bias")
 print(scale_lev1, scale_lev2, scale_lev3, bias_lev3, scale_lev4, bias_lev4, scale_lev5, bias_lev5, scale_lev6, bias_lev6, scale_lev7, bias_lev7, scale_lev8, bias_lev8)

 #quan_lev1 = np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F]/scale_lev1+bias_lev1)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F] = scale_lev1*(quan_lev1 - bias_lev1)
 #quan_lev2 = np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F]/scale_lev2+bias_lev2)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F] = scale_lev2*(quan_lev2 - bias_lev2)

 quan_lev1 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F]/scale_lev1),q_minimum,q_maximum)
 print(quan_lev1)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F] = quan_lev1
 newparams[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F] = scale_lev1*quan_lev1
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[0]*F:pointer+HASHMAP_OFFSET_TABLE[1]*F])

 quan_lev2 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F]/scale_lev2),q_minimum,q_maximum)
 print(quan_lev2)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F] = quan_lev2
 newparams[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F] = scale_lev2*quan_lev2
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[1]*F:pointer+HASHMAP_OFFSET_TABLE[2]*F])

 quan_lev3 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F]/scale_lev3),q_minimum,q_maximum)
 print(quan_lev3)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F] = quan_lev3
 newparams[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F] = scale_lev3*quan_lev3
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[2]*F:pointer+HASHMAP_OFFSET_TABLE[3]*F])

 quan_lev4 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F]/scale_lev4),q_minimum,q_maximum)
 print(quan_lev4)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F] = quan_lev4
 newparams[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F] = scale_lev4*quan_lev4
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[3]*F:pointer+HASHMAP_OFFSET_TABLE[4]*F])

 quan_lev5 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F]/scale_lev5),q_minimum,q_maximum)
 print(quan_lev5)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F] = quan_lev5
 newparams[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F] = scale_lev5*quan_lev5
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[4]*F:pointer+HASHMAP_OFFSET_TABLE[5]*F])

 quan_lev6 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F]/scale_lev6),q_minimum,q_maximum)
 print(quan_lev6)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F] = quan_lev6
 newparams[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F] = scale_lev6*quan_lev6
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[5]*F:pointer+HASHMAP_OFFSET_TABLE[6]*F])

 quan_lev7 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F]/scale_lev7),q_minimum,q_maximum)
 print(quan_lev7)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F] = quan_lev7
 newparams[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F] = scale_lev7*quan_lev7
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[6]*F:pointer+HASHMAP_OFFSET_TABLE[7]*F])

 quan_lev8 = np.clip(np.round(newparams[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F]/scale_lev8),q_minimum,q_maximum)
 print(quan_lev8)
 #newparams[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F] = quan_lev8
 newparams[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F] = scale_lev8*quan_lev8
 print(newparams[pointer+HASHMAP_OFFSET_TABLE[7]*F:pointer+HASHMAP_OFFSET_TABLE[8]*F])
 print(type(newparams[pointer]))
 encoding['params'] = newparams[pointer:pointer + HASHMAP_OFFSET_TABLE[-1]*F]
 #encoding['params'].tolist()
 # Debug log
 #print(f'F: {F}')
 #print(f'L: {L}')
 #print(f'T_max: {T_max}')
 #print(f'N_min: {N_min}')
 #print(f'N_max: {N_max}')
 #print(f'b: {b}')
 #for lev in range(L):
  #print(HASHMAP_OFFSET_TABLE[lev+1] - HASHMAP_OFFSET_TABLE[lev])
 return encoding, newparams, scales
# instant-ngp
def forward_encoding(mode, encoding, data_in):
 #print(f'[*] Hash Encoding Layer...')
 # 1. parse hyperparameters
 HASHMAP_OFFSET_TABLE = encoding['hashmap_offset_table']
 L = encoding['L']
 F = encoding['F']
 N_min = encoding['N_min']
 b = encoding['b']
 DIM = encoding['dim']
 NEIGHBORS = 2 ** DIM
 NUM_ELEMENTS = len(data_in)
 # 2. hash encoding layer
 encoding_params = np.reshape(encoding['params'], (-1, F))
 result = np.zeros(shape=(L*F*NUM_ELEMENTS,), dtype='float16')
 '''
 for i in range(NUM_ELEMENTS):
  pos_in = data_in[i]
  for lev in range(L):
 '''
 # per-level dataflow
 for lev in range(L):
  # setup knobs
  grid = HASHMAP_OFFSET_TABLE[lev]
  hashmap_size = HASHMAP_OFFSET_TABLE[lev + 1] - HASHMAP_OFFSET_TABLE[lev]
  scale = N_min * (b ** lev) - 1.0
  N_lev = 0
  tmp = int(math.ceil(scale))
  if (tmp & (tmp - 1)): 
   N_lev = tmp + 1
  else: 
   N_lev = tmp

  for i in range(NUM_ELEMENTS):
   pos_in = data_in[i]
   '''
   # setup knobs
   grid = HASHMAP_OFFSET_TABLE[lev]
   hashmap_size = HASHMAP_OFFSET_TABLE[lev + 1] - HASHMAP_OFFSET_TABLE[lev]
   scale = N_min * (b ** lev) - 1.0;
   N_lev = 0
   tmp = int(math.ceil(scale))
   if (tmp & (tmp - 1)): 
   N_lev = tmp + 1
   else: 
   N_lev = tmp
   '''
   encoded_feat = np.zeros(shape=(F,), dtype='float32')
   pos, pos_grid = pos_fract(pos_in, scale, DIM) # grid index of pos
   for idx in range(NEIGHBORS):
    weight = 1.0 # fp32
    pos_grid_local = np.zeros(shape=(DIM,), dtype='int32')

    for dim in range(DIM):
     if ((idx & (1 << dim)) == 0):
      weight *= 1.0 - pos[dim]
      pos_grid_local[dim] = pos_grid[dim]
     else:
      weight *= pos[dim]
      pos_grid_local[dim] = pos_grid[dim] + 1
    index = grid + grid_index(pos_grid_local, hashmap_size, N_lev, DIM)
    val = encoding_params[index]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    for f in range(F):
     data = val[f]
     encoded_feat[f] += weight * data

   for f in range(F):
    result[i + (lev * F + f) * NUM_ELEMENTS] = encoded_feat[f]

 result = np.reshape(result, (L*F, -1))
 #print(result.shape)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
 return result 

def forward_network(mode, network, data_in):
 #print(f'[*] MLP Layer...')

 # 1. parse hyperparameters
 input_width = network['input_width']    
 hidden_width = network['hidden_width']
 output_width = network['output_width']
 padded_output_width = network['padded_output_width']
 activation = network['activation']
 output_activation = network['output_activation']
 n_hidden_layers = network['n_hidden_layers']

 n_hidden_matmuls = 0
 if n_hidden_layers > 0: 
  n_hidden_matmuls = n_hidden_layers - 1

 network_params = network['params'] # FIXME: a list of numpy arrays

 # 2. mlp computation
 result = data_in
 if n_hidden_layers == 0:
  # output layer
  weight = np.reshape(network_params[-1], (padded_output_width, input_width))
  result = fc_layer(weight, result) # slice
  result = act_layer(output_activation, result)
 else:
 # input layer
  weight = np.reshape(network_params[0], (hidden_width, input_width))
  result = fc_layer(weight, result)
  result = act_layer(activation, result)    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
  # hidden layers
  for i in range(n_hidden_matmuls):
   weight = np.reshape(network_params[i+1], (hidden_width, hidden_width))
   result = fc_layer(weight, result)
   result = act_layer(activation, result)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
  # output layer
  weight = np.reshape(network_params[-1], (padded_output_width, hidden_width))
  result = fc_layer(weight, result)
  result = act_layer(activation, result)

 result = result[:output_width,:] # slice
 return result

#### helper_functions ####
# hash encoding
# return grid index of input pos
def pos_fract(in_pos, scale, dim):
 pos = in_pos * scale + 0.5
 tmp = np.floor(pos) 
 pos_grid = tmp.astype('int32')
 pos -= tmp.astype('float32')

 return pos, pos_grid

# lookup a single Hash Table Entry
def grid_index(pos_grid_local, hashmap_size, N_lev, dim):
 stride = 1
 index = 0
 for d in range(dim):
  index += pos_grid_local[d] * stride
  stride *= N_lev

 # check if there is a collision
 if hashmap_size < stride:
  index = fast_hash(pos_grid_local, dim)

 index = index % hashmap_size 
 return index

def fast_hash(pos_grid, dim):
 PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
 result = 0
 for i in range(dim):
  result ^= pos_grid[i] * PRIMES[i]
 return result 

# mlp
def act_layer(mode, data_in):
 if mode == 'ReLU':
  data_out = np.maximum(0, data_in)
 else: # None
  data_out = data_in
 return data_out

def fc_layer(weight, data_in):
 data_out = np.dot(weight, data_in)
 return data_out 

#### main ####
def main():
# load traces
 #pos_fpath = f'./image/pos-file.txt'
 #color_fpath = f'./image/color-file.txt'

 #positions = np.loadtxt(pos_fpath, dtype='float32')
 #colors = np.loadtxt(color_fpath, dtype='float32')

 # load model
 model_fpath = f'./fox.msgpack' 
 config = utils.load_msgpack(model_fpath)
 params_bin = config['snapshot']['params_binary']
 print(config['snapshot']['params_type'])
 print(params_bin[-30:])
 # FIXME: should explicitly change precision
 params = np.frombuffer(params_bin, dtype='float16') # FIXME: native byte ordering
 mode = 'nerf'
 model, newparams, scales = init_model(mode, config, params)
 config['snapshot']['params_binary']=newparams.tobytes()
 #config['snapshot']['scale_factors']=scales
 #print("Here is scales!")
 print(scales)
 print(config['snapshot']['params_binary'][-30:])
 #print(params[-30:])
 #print(np.frombuffer(newparams.tobytes(),dtype='float16')[-30:])
 #model['density_network']['params'][0]=model['density_network']['params'][0].tolist()
 #model['density_network']['params'][1]=model['density_network']['params'][1].tolist()
 #model['rgb_network']['params'][0]=model['rgb_network']['params'][0].tolist()
 #model['rgb_network']['params'][1]=model['rgb_network']['params'][1].tolist()
 #model['rgb_network']['params'][2]=model['rgb_network']['params'][2].tolist()
 #model['pos_encoding']['params']=model['pos_encoding']['params'].tolist()
 #for v in model.values():
  #print(v)

 #q_model_fpath = f'./fox_temp.msgpack' #다 dequantize해서 돌려줌
 q_model_fpath = f'./fox_int8.msgpack'  #hash는 dequantize해서 돌려주고, mlp는 int로
 #q_model_fpath = f'./fox_temp_int4.msgpack'
 #q_model_fpath = f'./fox_new.msgpack'

 utils.save_msgpack(q_model_fpath, config)
 new_config = utils.load_msgpack(q_model_fpath)
 #print(config['snapshot']['scale_factors'])
 #print(config['snapshot']['scale_factors'][0])
 #print(np.array_equal(newparams,np.frombuffer(config['snapshot']['params_binary'], dtype='float16')))
 #utils.save_msgpack(q_model_fpath,model)
 #result = forward_encoding(mode, model['encoding'], positions)
 # checkpoint
 #np.save("hash_enc", result)
 #result = forward_network(mode, model['network'], result)
 #result = result.T

 # validate
 #reference = colors

 #diff = reference - result
 #mse = (np.square(diff)).mean()

if __name__=='__main__':
 main()
