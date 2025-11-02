import pickle
import numpy as np
import scipy.io as sio

# Check baseline data
baseline = np.load('artifacts/baseline_X.npy')
print('Baseline shape:', baseline.shape)

# Check edge data
edge_index = sio.loadmat('artifacts/edge_index.mat')['edge_index']
edge_attr = sio.loadmat('artifacts/edge_attr.mat')['edge_attr']
print('Edge index shape:', edge_index.shape)
print('Edge attr shape:', edge_attr.shape)
print('Edge index sample:', edge_index[:5])

# Check scalers
with open('artifacts/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
print('Scalers keys:', list(scalers.keys()))

# Check if we have the metadata
if 'num_nodes' in scalers:
    print('Num nodes:', scalers['num_nodes'])
if 'in_dim' in scalers:
    print('In dim:', scalers['in_dim'])
if 'out_dim' in scalers:
    print('Out dim:', scalers['out_dim'])
if 'edge_dim' in scalers:
    print('Edge dim:', scalers['edge_dim'])
