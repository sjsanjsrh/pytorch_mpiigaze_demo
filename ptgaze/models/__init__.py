import importlib
import logging

import timm
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _get_device(config: DictConfig):
    """DirectML 지원을 포함한 디바이스 반환"""
    if config.device == 'dml':
        import torch_directml
        return torch_directml.device()
    else:
        return torch.device(config.device)


def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    if mode in ['MPIIGaze', 'MPIIFaceGaze']:
        module = importlib.import_module(
            f'ptgaze.models.{mode.lower()}.{config.model.name}')
        model = module.Model(config)
    elif mode == 'ETH-XGaze':
        model = timm.create_model(config.model.name, num_classes=2)
    else:
        raise ValueError
    device = _get_device(config)
    model.to(device)
    return model
