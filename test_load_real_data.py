"""
Test script to load real IEEE dataset and check structure
"""

import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path

def load_mat_file(filepath):
    """
    Load .mat file (handles both v7 and v7.3 formats)
    """
    try:
        # Try scipy first (for MATLAB v7)
        data = sio.loadmat(filepath)
        print(f"✅ Loaded with scipy.io: {filepath.name}")
        return data
    except NotImplementedError:
        # Fall back to h5py (for MATLAB v7.3)
        print(f"🔄 Trying h5py for: {filepath.name}")
        with h5py.File(filepath, 'r') as f:
            data = {}
            for key in f.keys():
                if key.startswith('__'):
                    continue
                try:
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        data[key] = np.array(item)
                    print(f"   Loaded key: {key}, shape: {data[key].shape if key in data else 'N/A'}")
                except Exception as e:
                    print(f"   ⚠️ Could not load key {key}: {e}")
            return data

def explore_ieee_dataset(bus_system='ieee39'):
    """
    Explore IEEE dataset structure
    """
    
    print("=" * 70)
    print(f"🔍 Exploring {bus_system.upper()} Dataset")
    print("=" * 70)
    
    data_dir = Path(f'dataset_pf_opf/{bus_system}/{bus_system}/raw')
    
    if not data_dir.exists():
        print(f"❌ Dataset not found at {data_dir}")
        return
    
    print(f"\n📁 Dataset directory: {data_dir}")
    print(f"📄 Files found:")
    for f in sorted(data_dir.glob('*.mat')):
        print(f"   - {f.name}")
    
    # Load each file
    print("\n" + "=" * 70)
    print("📊 Loading Files...")
    print("=" * 70)
    
    files_to_load = {
        'Xopf.mat': 'Input features (OPF)',
        'Y_polar_opf.mat': 'Output labels (OPF)',
        'edge_index_opf.mat': 'Graph edges',
        'edge_attr_opf.mat': 'Edge features'
    }
    
    loaded_data = {}
    
    for filename, description in files_to_load.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n⚠️ {filename} not found")
            continue
        
        print(f"\n📄 {filename} - {description}")
        print("-" * 70)
        
        try:
            data = load_mat_file(filepath)
            
            # Print all keys
            print(f"   Keys in file: {[k for k in data.keys() if not k.startswith('__')]}")
            
            # Try to find the main data
            for key in data.keys():
                if key.startswith('__'):
                    continue
                
                value = data[key]
                print(f"\n   Key: '{key}'")
                print(f"   Type: {type(value)}")
                print(f"   Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                print(f"   Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
                
                if hasattr(value, 'shape') and len(value.shape) > 0:
                    print(f"   Min: {np.min(value) if np.issubdtype(value.dtype, np.number) else 'N/A'}")
                    print(f"   Max: {np.max(value) if np.issubdtype(value.dtype, np.number) else 'N/A'}")
                    print(f"   Sample (first 5): {value.flatten()[:5]}")
                
                loaded_data[filename] = {key: value}
        
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Dataset Summary")
    print("=" * 70)
    
    if loaded_data:
        print("\n✅ Successfully loaded files:")
        for filename in loaded_data.keys():
            print(f"   - {filename}")
        
        # Try to infer dataset structure
        print("\n🔍 Inferring Dataset Structure:")
        
        # Check Xopf for number of samples and features
        if 'Xopf.mat' in loaded_data:
            xopf_data = loaded_data['Xopf.mat']
            for key, value in xopf_data.items():
                if hasattr(value, 'shape'):
                    print(f"\n   Input Features (Xopf/{key}):")
                    print(f"   Shape: {value.shape}")
                    if len(value.shape) == 2:
                        print(f"   → Samples: {value.shape[0]}")
                        print(f"   → Features per sample: {value.shape[1]}")
        
        # Check Y_polar_opf for outputs
        if 'Y_polar_opf.mat' in loaded_data:
            y_data = loaded_data['Y_polar_opf.mat']
            for key, value in y_data.items():
                if hasattr(value, 'shape'):
                    print(f"\n   Output Labels (Y_polar_opf/{key}):")
                    print(f"   Shape: {value.shape}")
                    if len(value.shape) == 2:
                        print(f"   → Samples: {value.shape[0]}")
                        print(f"   → Outputs per sample: {value.shape[1]}")
        
        # Check edge_index
        if 'edge_index_opf.mat' in loaded_data:
            edge_data = loaded_data['edge_index_opf.mat']
            for key, value in edge_data.items():
                if hasattr(value, 'shape'):
                    print(f"\n   Graph Edges (edge_index_opf/{key}):")
                    print(f"   Shape: {value.shape}")
                    print(f"   → Number of edges: {value.shape[1] if len(value.shape) > 1 else value.shape[0]}")
        
        # Check edge_attr
        if 'edge_attr_opf.mat' in loaded_data:
            edge_attr_data = loaded_data['edge_attr_opf.mat']
            for key, value in edge_attr_data.items():
                if hasattr(value, 'shape'):
                    print(f"\n   Edge Features (edge_attr_opf/{key}):")
                    print(f"   Shape: {value.shape}")
    
    else:
        print("\n❌ No files loaded successfully")
    
    print("\n" + "=" * 70)
    print("✅ Exploration Complete!")
    print("=" * 70)

if __name__ == "__main__":
    explore_ieee_dataset('ieee39')
