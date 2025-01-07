"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
"""

import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from .utils import IntermediateLayerGetter
from ...core import register

import torch.nn as nn 
import torch.nn.functional as F
from .common import get_activation, FrozenBatchNorm2d


@register()
class TimmModel(torch.nn.Module):
    def __init__(self, \
        name, 
        return_layers, 
        pretrained_path="",
        pretrained=False,
        exportable=True, 
        features_only=True,
        freeze_at=-1, 
        freeze_norm=True, 
        **kwargs) -> None:

        super().__init__()

        import timm
        model = timm.create_model(
            name,
            pretrained=pretrained, 
            exportable=exportable, 
            features_only=features_only,
            **kwargs
        )
        # nodes, _ = get_graph_node_names(model)
        # print(nodes)
        # features = {'': ''}
        # model = create_feature_extractor(model, return_nodes=features)

        assert set(return_layers).issubset(model.feature_info.module_name()), \
            f'return_layers should be a subset of {model.feature_info.module_name()}'
        
        # self.model = model
        self.model = IntermediateLayerGetter(model, return_layers)

        return_idx = [model.feature_info.module_name().index(name) for name in return_layers]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers

        if freeze_at >= 0:
            self._freeze_parameters(self.model.conv1)
            self._freeze_parameters(self.model.bn1)

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained_path != "":
            pretrained_weights = torch.load(pretrained_path)["state_dict"]
            new_state_dict = {}
            for k, v in pretrained_weights.items():
                new_k = f"model.{k}"
                new_state_dict[new_k] = v
            self.load_state_dict(new_state_dict, strict=False)
            print(f'Load {len(new_state_dict)} keys')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x: torch.Tensor): 
        outputs = self.model(x)
        # outputs = [outputs[i] for i in self.return_idx]
        return outputs


if __name__ == '__main__':
    
    model = TimmModel(name='resnet34', return_layers=['layer2', 'layer3'])
    data = torch.rand(1, 3, 640, 640)
    outputs = model(data)
    
    for output in outputs:
        print(output.shape)

    """
    model:
        type: TimmModel
        name: resnet34
        return_layers: ['layer2', 'layer4']
    """
