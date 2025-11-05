"""입술 회귀 모델 아키텍처 및 파라미터 수 출력 스크립트"""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch
import numpy as np


def count_params(model: torch.nn.Module):
    """모델의 전체/학습 가능 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_layer_info(model: torch.nn.Module, indent=0):
    """재귀적으로 모델 레이어 구조 출력"""
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{'  ' * indent}{name}: {module.__class__.__name__} ({num_params:,} params)")
        if len(list(module.children())) > 0:
            print_layer_info(module, indent + 1)


def main():
    model_path = Path("artifacts/models/mouth_regressor.pt")
    
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("Please train the model first using train_mouth_regressor.py")
        return
    
    print("=" * 60)
    print("Mouth Regressor Model Analysis")
    print("=" * 60)
    
    # Load model bundle
    bundle = torch.load(model_path, map_location='cpu', weights_only=False)
    config = bundle['config']
    state_dict = bundle['state_dict']
    
    print(f"\n📁 Model Path: {model_path}")
    print(f"📦 File Size: {model_path.stat().st_size / 1024:.2f} KB")
    
    print("\n⚙️  Configuration:")
    print(f"  • Input dim: {config['input_dim']}")
    print(f"  • Hidden dim: {config['hidden_dim']}")
    print(f"  • Dropout: {config['dropout']}")
    print(f"  • Model type: {config.get('model_type', 'unknown')}")
    if config.get('indices') is not None:
        print(f"  • Selected landmarks: {len(config['indices'])} indices")
    print(f"  • Landmark dims: {config.get('dims', 2)}")
    
    # Reconstruct model
    from train_mouth_regressor import MouthRegressor
    model = MouthRegressor(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    model.load_state_dict(state_dict)
    
    total, trainable = count_params(model)
    
    print(f"\n📊 Parameters:")
    print(f"  • Total: {total:,} params")
    print(f"  • Trainable: {trainable:,} params")
    print(f"  • Model size (float32): ~{total * 4 / 1024:.2f} KB")
    
    print("\n🏗️  Architecture:")
    print_layer_info(model)
    
    print("\n📈 Layer Details:")
    for name, param in model.named_parameters():
        print(f"  {name}: {tuple(param.shape)} = {param.numel():,} params")
    
    # Test forward pass
    print("\n✅ Forward Pass Test:")
    dummy_input = torch.randn(1, config['input_dim'])
    with torch.no_grad():
        openness, lip_position = model(dummy_input)
    print(f"  Input shape: {tuple(dummy_input.shape)}")
    print(f"  Output (openness): {tuple(openness.shape)} → range [0, 1]")
    print(f"  Output (lip_position): {tuple(lip_position.shape)} → range [-1, 1]")
    
    print("\n" + "=" * 60)
    print("비교:")
    print(f"  • MPIIGaze (Eye gaze): 77,046 params (~0.29 MB)")
    print(f"  • MPIIFaceGaze (Face gaze): 2,883,395 params (~11 MB)")
    print(f"  • MouthRegressor: {total:,} params (~{total * 4 / 1024:.2f} KB)")
    print(f"  → 입술 모델은 MPIIGaze의 {total / 77046 * 100:.1f}% 크기")
    print("=" * 60)


if __name__ == '__main__':
    main()
