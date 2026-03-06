import torch
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from components.component_4.model import MultiTaskNet

def test_model_loading():
    model_path = "experiments/comp4_experiments/final_multitask_model.pth"
    config_path = "experiments/comp4_experiments/model_config.json"
    
    print("="*70)
    print("TESTING MODEL LOADING WITH CONFIGURATION")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("   Using default configuration...")
        config = {}
    else:
        print(f"Found model: {model_path}")
        print(f"Found config: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    print("\n--- Configuration ---")
    print(f"  Shared Layers: {config.get('shared_layers', [256, 128])}")
    print(f"  Head Hidden:   {config.get('head_hidden', 64)}")
    print(f"  Head Depth:    {config.get('head_depth', 1)}")
    print(f"  Dropout:       {config.get('dropout', 0.2)}")
    
    try:
        model = MultiTaskNet(
            input_dim=19,
            shared_layers=config.get("shared_layers", [256, 128]),
            head_hidden=config.get("head_hidden", 64),
            head_depth=config.get("head_depth", 1),
            dropout=config.get("dropout", 0.2)
        )
        print("\nModel architecture created successfully")
    except Exception as e:
        print(f"\nError creating model: {e}")
        return False
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("Weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False
    
    try:
        model.eval()
        dummy_input = torch.randn(1, 19)
        with torch.no_grad():
            pred_htn, pred_hf, pred_cluster = model(dummy_input)
        
        print("\n--- Test Inference ---")
        print(f"  HTN prediction:     {pred_htn.item():.4f}")
        print(f"  HF prediction:      {pred_hf.item():.4f}")
        print(f"  Cluster logits:     {pred_cluster.numpy()[0]}")
        print("\nModel inference working correctly")
    except Exception as e:
        print(f"\nError during inference: {e}")
        return False
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED - Dashboard will load model correctly")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
