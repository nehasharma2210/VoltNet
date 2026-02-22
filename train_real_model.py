"""
Train VoltNet model on real IEEE dataset
Supports: IEEE24, IEEE39, IEEE118, UK, Texas
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm
import json

# Import model
import sys
sys.path.append('backend')
from model import SurrogateGNN

def load_real_ieee_data(bus_system='ieee39', task='OPF'):
    """
    Load REAL IEEE dataset with proper h5py reference handling
    """
    import h5py
    from pathlib import Path
    
    print(f"📥 Loading REAL {bus_system.upper()} {task} dataset...")
    
    data_dir = Path(f'dataset_pf_opf/{bus_system}/{bus_system}/raw')
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {data_dir}")
    
    def load_mat_with_refs(filepath):
        """Load .mat file and dereference h5py objects"""
        try:
            # Try scipy first
            import scipy.io as sio
            data = sio.loadmat(filepath)
            result = {k: v for k, v in data.items() if not k.startswith('__')}
            return result
        except NotImplementedError:
            # Use h5py for v7.3 format
            with h5py.File(filepath, 'r') as f:
                result = {}
                for key in f.keys():
                    if key.startswith('__') or key == '#refs#':
                        continue
                    
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        data = np.array(item)
                        
                        # Dereference object references
                        if data.dtype == np.dtype('O'):
                            dereferenced = []
                            for ref in data.flatten():
                                if isinstance(ref, h5py.h5r.Reference):
                                    obj = f[ref]
                                    if isinstance(obj, h5py.Dataset):
                                        dereferenced.append(np.array(obj))
                            if dereferenced:
                                result[key] = np.array(dereferenced)
                        else:
                            result[key] = data
                return result
    
    # Load files
    print("   Loading Xopf.mat...")
    X_data = load_mat_with_refs(data_dir / 'Xopf.mat')
    X = X_data['X']
    
    print("   Loading Y_polar_opf.mat...")
    Y_data = load_mat_with_refs(data_dir / 'Y_polar_opf.mat')
    Y = Y_data['Y_polar']
    
    print("   Loading edge_index_opf.mat...")
    edge_index_data = load_mat_with_refs(data_dir / 'edge_index_opf.mat')
    edge_index = edge_index_data['edge_index']
    
    print("   Loading edge_attr_opf.mat...")
    edge_attr_data = load_mat_with_refs(data_dir / 'edge_attr_opf.mat')
    edge_attr = edge_attr_data['edge_attr']
    
    # Transpose if needed (MATLAB format: samples, features, nodes)
    if len(X.shape) == 3 and X.shape[1] < X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
        Y = np.transpose(Y, (0, 2, 1))
    
    # Convert edge_index to 0-indexed
    if edge_index.min() == 1:
        edge_index = edge_index - 1
    
    num_samples = X.shape[0]
    num_nodes = X.shape[1]
    num_features = X.shape[2]
    output_dim = Y.shape[2]
    
    print(f"✅ Loaded REAL data:")
    print(f"   Samples: {num_samples}")
    print(f"   Nodes: {num_nodes}")
    print(f"   Features: {num_features}")
    print(f"   Output dim: {output_dim}")
    
    return X, Y, edge_index, edge_attr, num_nodes, num_features, output_dim


def create_dummy_dataset(num_samples=900, num_nodes=39, num_features=4, output_dim=3):
    """
    Create dummy dataset for testing
    Replace this with actual IEEE data loading
    """
    
    print(f"📊 Creating dummy dataset for testing...")
    print(f"   Samples: {num_samples}")
    print(f"   Nodes: {num_nodes}")
    print(f"   Features: {num_features}")
    
    dataset = []
    
    # Create simple graph structure
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    edge_attr = torch.randn(edge_index.size(1), 2)
    
    # Create samples
    for _ in range(num_samples):
        x = torch.randn(num_nodes, num_features)
        y = torch.randn(num_nodes, output_dim)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        dataset.append(data)
    
    return dataset

def train_model(args):
    """
    Train VoltNet on IEEE dataset
    """
    
    print("=" * 70)
    print(f"🚀 Training VoltNet on {args.bus_system.upper()} Dataset")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load dataset
    print("\n📥 Loading dataset...")
    
    try:
        # Try loading REAL IEEE data
        X, Y, edge_index, edge_attr, num_nodes, num_features, output_dim = load_real_ieee_data(
            bus_system=args.bus_system, 
            task=args.task
        )
        
        # Use subset for faster training (optional)
        if args.max_samples > 0 and args.max_samples < len(X):
            print(f"\n⚡ Using subset of {args.max_samples} samples for faster training")
            indices = np.random.choice(len(X), args.max_samples, replace=False)
            X = X[indices]
            Y = Y[indices]
        
        # 🔥 NORMALIZE DATA for better training
        print(f"\n📊 Normalizing data...")
        from sklearn.preprocessing import StandardScaler
        
        # Normalize X (inputs)
        X_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        scaler_x = StandardScaler()
        X_normalized = scaler_x.fit_transform(X_flat).reshape(X_shape)
        
        # Normalize Y (outputs)
        Y_shape = Y.shape
        Y_flat = Y.reshape(-1, Y.shape[-1])
        scaler_y = StandardScaler()
        Y_normalized = scaler_y.fit_transform(Y_flat).reshape(Y_shape)
        
        # Normalize edge_attr
        scaler_e = StandardScaler()
        edge_attr_normalized = scaler_e.fit_transform(edge_attr)
        
        print(f"   ✅ X normalized: mean={X_normalized.mean():.4f}, std={X_normalized.std():.4f}")
        print(f"   ✅ Y normalized: mean={Y_normalized.mean():.4f}, std={Y_normalized.std():.4f}")
        
        # Create PyTorch Geometric dataset
        dataset = []
        edge_index_t = torch.tensor(edge_index.T, dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr_normalized, dtype=torch.float)
        
        print(f"\n📊 Creating PyTorch Geometric dataset...")
        for i in tqdm(range(len(X_normalized)), desc="Processing samples"):
            x = torch.tensor(X_normalized[i], dtype=torch.float)
            y = torch.tensor(Y_normalized[i], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y)
            dataset.append(data)
        
        metadata = {
            'num_nodes': num_nodes,
            'node_features': num_features,
            'edge_features': edge_attr.shape[1],
            'output_dim': output_dim,
            'num_samples': len(X_normalized),
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'scaler_e': scaler_e
        }
        
        print(f"✅ Using REAL IEEE data!")
        
    except Exception as e:
        print(f"\n⚠️ Could not load real data: {e}")
        print(f"   Falling back to dummy data for testing...")
        
        # Fallback to dummy data
        dataset = create_dummy_dataset(num_samples=900, num_nodes=39, num_features=4, output_dim=3)
        
        metadata = {
            'num_nodes': 39,
            'node_features': 4,
            'edge_features': 2,
            'output_dim': 3,
            'num_samples': 900
        }
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = SurrogateGNN(
        in_dim=metadata['node_features'],
        edge_dim=metadata['edge_features'],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        out_node_dim=metadata['output_dim']
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\n🏋️ Training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metadata': metadata
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"   ✅ Saved best model (val_loss: {val_loss:.6f})")
    
    print("=" * 70)
    print(f"✅ Training complete!")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    
    # Test evaluation
    print(f"\n🧪 Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), weights_only=False)['model_state_dict'])
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f"   Test Loss: {test_loss:.6f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
        'metadata': {
            'num_nodes': metadata['num_nodes'],
            'node_features': metadata['node_features'],
            'edge_features': metadata['edge_features'],
            'output_dim': metadata['output_dim'],
            'num_samples': metadata['num_samples']
        },
        'args': vars(args)
    }
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Model saved to: {args.save_dir}/best_model.pth")
    print(f"📊 History saved to: {args.save_dir}/training_history.json")
    
    return model, history


def prepare_artifacts_for_deployment(args, metadata):
    """
    Prepare artifacts for backend deployment
    """
    
    print("\n📦 Preparing artifacts for deployment...")
    
    artifact_dir = 'backend/artifacts'
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Copy trained model
    import shutil
    shutil.copy(
        os.path.join(args.save_dir, 'best_model.pth'),
        os.path.join(artifact_dir, 'best_model.pth')
    )
    print(f"   ✅ Copied best_model.pth")
    
    # Save scalers (use real scalers if available)
    if 'scaler_x' in metadata and 'scaler_y' in metadata and 'scaler_e' in metadata:
        # Use real scalers from training
        scalers_data = {
            'scaler_x': metadata['scaler_x'],
            'scaler_y': metadata['scaler_y'],
            'scaler_e': metadata['scaler_e'],
            'num_nodes': metadata['num_nodes'],
            'in_dim': metadata['node_features'],
            'out_dim': metadata['output_dim'],
            'edge_dim': metadata['edge_features']
        }
        print(f"   ✅ Using REAL scalers from training data")
    else:
        # Create dummy scalers (fallback)
        from sklearn.preprocessing import MinMaxScaler
        
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_e = MinMaxScaler()
        
        num_nodes = metadata['num_nodes']
        in_dim = metadata['node_features']
        out_dim = metadata['output_dim']
        edge_dim = metadata['edge_features']
        
        dummy_x = np.random.randn(1000, num_nodes * in_dim)
        dummy_y = np.random.randn(1000, num_nodes * out_dim)
        dummy_e = np.random.randn(100, edge_dim)
        
        scaler_x.fit(dummy_x)
        scaler_y.fit(dummy_y)
        scaler_e.fit(dummy_e)
        
        scalers_data = {
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'scaler_e': scaler_e,
            'num_nodes': num_nodes,
            'in_dim': in_dim,
            'out_dim': out_dim,
            'edge_dim': edge_dim
        }
        print(f"   ⚠️ Using dummy scalers (fallback)")
    
    with open(os.path.join(artifact_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers_data, f)
    print(f"   ✅ Created scalers.pkl")
    
    # Create baseline_X
    baseline_X = np.ones((metadata['num_nodes'], metadata['node_features']))
    np.save(os.path.join(artifact_dir, 'baseline_X.npy'), baseline_X)
    print(f"   ✅ Created baseline_X.npy")
    
    print(f"\n✅ Artifacts ready for deployment in: {artifact_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VoltNet on real IEEE dataset')
    
    # Dataset args
    parser.add_argument('--bus_system', type=str, default='ieee39',
                        choices=['ieee24', 'ieee39', 'ieee118', 'uk', 'texas'],
                        help='IEEE bus system to use')
    parser.add_argument('--task', type=str, default='OPF',
                        choices=['PF', 'OPF'],
                        help='Task: Power Flow or Optimal Power Flow')
    parser.add_argument('--data_root', type=str, default='dataset_pf_opf',
                        help='Root directory containing dataset')
    
    # Model args
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum samples to use (0 = use all, default=5000 for faster training)')
    
    # Save args
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train model
    try:
        model, history = train_model(args)
        
        # Prepare artifacts for deployment
        prepare_artifacts_for_deployment(args, history['metadata'])
        
        print("\n" + "=" * 70)
        print("🎉 Training pipeline complete!")
        print("=" * 70)
        print("\n📝 Next steps:")
        print("1. Start backend: cd backend && python app.py")
        print("2. Open frontend: http://localhost:8000")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
