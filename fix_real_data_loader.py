"""
Fixed loader for IEEE dataset with proper h5py reference handling
"""

import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path

def load_mat_with_references(filepath):
    """
    Load .mat file and dereference h5py object references
    """
    try:
        # Try scipy first
        data = sio.loadmat(filepath)
        print(f"✅ Loaded with scipy: {filepath.name}")
        
        # Extract main data (skip __ keys)
        result = {}
        for key in data.keys():
            if not key.startswith('__'):
                result[key] = data[key]
        return result
        
    except NotImplementedError:
        # Use h5py for v7.3 format
        print(f"🔄 Using h5py for: {filepath.name}")
        
        with h5py.File(filepath, 'r') as f:
            result = {}
            
            for key in f.keys():
                if key.startswith('__') or key == '#refs#':
                    continue
                
                item = f[key]
                
                if isinstance(item, h5py.Dataset):
                    data = np.array(item)
                    
                    # Check if it's object references
                    if data.dtype == np.dtype('O'):
                        print(f"   📦 Dereferencing {key}...")
                        
                        # Dereference each object
                        dereferenced = []
                        for i, ref in enumerate(data.flatten()):
                            if isinstance(ref, h5py.h5r.Reference):
                                try:
                                    obj = f[ref]
                                    if isinstance(obj, h5py.Dataset):
                                        dereferenced.append(np.array(obj))
                                except Exception as e:
                                    print(f"      ⚠️ Could not dereference index {i}: {e}")
                        
                        if dereferenced:
                            # Stack all dereferenced arrays
                            result[key] = np.array(dereferenced)
                            print(f"   ✅ Dereferenced {key}: shape {result[key].shape}")
                        else:
                            print(f"   ⚠️ No valid references found in {key}")
                            result[key] = data
                    else:
                        result[key] = data
                        print(f"   ✅ Loaded {key}: shape {data.shape}")
            
            return result

def load_ieee_dataset_real(bus_system='ieee39', task='OPF'):
    """
    Load real IEEE dataset with proper reference handling
    """
    
    print("=" * 70)
    print(f"🚀 Loading {bus_system.upper()} {task} Dataset (REAL DATA)")
    print("=" * 70)
    
    data_dir = Path(f'dataset_pf_opf/{bus_system}/{bus_system}/raw')
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {data_dir}")
    
    # Load files
    print("\n📥 Loading dataset files...")
    
    # Input features
    print("\n1️⃣ Loading input features (Xopf.mat)...")
    X_data = load_mat_with_references(data_dir / 'Xopf.mat')
    X = X_data['X']
    print(f"   Shape: {X.shape}")
    
    # Output labels
    print("\n2️⃣ Loading output labels (Y_polar_opf.mat)...")
    Y_data = load_mat_with_references(data_dir / 'Y_polar_opf.mat')
    Y = Y_data['Y_polar']
    print(f"   Shape: {Y.shape}")
    
    # Edge index
    print("\n3️⃣ Loading graph edges (edge_index_opf.mat)...")
    edge_index_data = load_mat_with_references(data_dir / 'edge_index_opf.mat')
    edge_index = edge_index_data['edge_index']
    print(f"   Shape: {edge_index.shape}")
    print(f"   Edges: {edge_index.shape[0]}")
    
    # Edge attributes
    print("\n4️⃣ Loading edge features (edge_attr_opf.mat)...")
    edge_attr_data = load_mat_with_references(data_dir / 'edge_attr_opf.mat')
    edge_attr = edge_attr_data['edge_attr']
    print(f"   Shape: {edge_attr.shape}")
    
    # Analyze structure
    print("\n" + "=" * 70)
    print("📊 Dataset Structure Analysis")
    print("=" * 70)
    
    num_samples = X.shape[0]
    print(f"\n✅ Number of samples: {num_samples}")
    
    # Infer number of nodes
    if len(X.shape) == 3:
        # Check if data is transposed (common in MATLAB)
        # IEEE39 should have 39 nodes, not 4
        if X.shape[1] < X.shape[2]:
            # Likely transposed: (samples, features, nodes) instead of (samples, nodes, features)
            print(f"⚠️ Data appears transposed: {X.shape}")
            print(f"   Transposing to (samples, nodes, features)...")
            X = np.transpose(X, (0, 2, 1))
            Y = np.transpose(Y, (0, 2, 1))
            print(f"   ✅ New X shape: {X.shape}")
            print(f"   ✅ New Y shape: {Y.shape}")
        
        num_nodes = X.shape[1]
        num_features = X.shape[2]
        print(f"✅ Nodes per sample: {num_nodes}")
        print(f"✅ Features per node: {num_features}")
    elif len(X.shape) == 2:
        # Need to infer from edge_index
        num_nodes = int(edge_index.max())
        num_features = X.shape[1] // num_nodes if X.shape[1] % num_nodes == 0 else X.shape[1]
        print(f"✅ Inferred nodes: {num_nodes}")
        print(f"✅ Inferred features per node: {num_features}")
        
        # Reshape X
        if X.shape[1] == num_nodes * num_features:
            X = X.reshape(num_samples, num_nodes, num_features)
            print(f"✅ Reshaped X to: {X.shape}")
    
    # Analyze Y
    if len(Y.shape) == 3:
        output_dim = Y.shape[2]
        print(f"✅ Output dimension: {output_dim}")
    elif len(Y.shape) == 2:
        output_dim = Y.shape[1] // num_nodes if Y.shape[1] % num_nodes == 0 else Y.shape[1]
        print(f"✅ Inferred output dimension: {output_dim}")
        
        # Reshape Y
        if Y.shape[1] == num_nodes * output_dim:
            Y = Y.reshape(num_samples, num_nodes, output_dim)
            print(f"✅ Reshaped Y to: {Y.shape}")
    
    # Edge info
    num_edges = edge_index.shape[0]
    edge_dim = edge_attr.shape[1] if len(edge_attr.shape) > 1 else 1
    print(f"✅ Number of edges: {num_edges}")
    print(f"✅ Edge features: {edge_dim}")
    
    # Convert edge_index to 0-indexed (MATLAB uses 1-indexed)
    if edge_index.min() == 1:
        edge_index = edge_index - 1
        print(f"✅ Converted edge_index to 0-indexed")
    
    # Sample statistics
    print("\n" + "=" * 70)
    print("📈 Data Statistics")
    print("=" * 70)
    
    print(f"\nInput Features (X):")
    print(f"   Min: {X.min():.4f}")
    print(f"   Max: {X.max():.4f}")
    print(f"   Mean: {X.mean():.4f}")
    print(f"   Std: {X.std():.4f}")
    
    print(f"\nOutput Labels (Y):")
    print(f"   Min: {Y.min():.4f}")
    print(f"   Max: {Y.max():.4f}")
    print(f"   Mean: {Y.mean():.4f}")
    print(f"   Std: {Y.std():.4f}")
    
    print(f"\nEdge Features:")
    print(f"   Min: {edge_attr.min():.4f}")
    print(f"   Max: {edge_attr.max():.4f}")
    
    # Return dataset
    dataset = {
        'X': X,
        'Y': Y,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }
    
    metadata = {
        'num_samples': num_samples,
        'num_nodes': num_nodes,
        'node_features': num_features,
        'output_dim': output_dim,
        'num_edges': num_edges,
        'edge_features': edge_dim,
        'bus_system': bus_system,
        'task': task
    }
    
    print("\n" + "=" * 70)
    print("✅ Dataset Loaded Successfully!")
    print("=" * 70)
    
    return dataset, metadata

if __name__ == "__main__":
    try:
        dataset, metadata = load_ieee_dataset_real('ieee39', 'OPF')
        
        print("\n🎉 SUCCESS!")
        print(f"\n📊 Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        print(f"\n💾 Dataset keys: {list(dataset.keys())}")
        print(f"\n✅ Ready for training!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
