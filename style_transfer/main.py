import os
from typing import List, Tuple, Dict, Optional
from functools import cached_property
from PIL.Image import Image

import torch
import torchvision
from torch import nn


class IO:

    def __init__(
        self, 
        content_img_path: os.PathLike, 
        style_img_path: os.PathLike, 
    ) -> None:
        self.content_img_path: os.PathLike = content_img_path
        self.style_img_path: os.PathLike = style_img_path

    @cached_property
    def content_tensor(self) -> torch.Tensor:
        return self.__to_tensor(path=self.content_img_path, target_size=None)
        
    @cached_property
    def style_tensor(self) -> torch.Tensor:
        return self.__to_tensor(path=self.style_img_path, target_size=self.content_tensor.shape)

    @cached_property
    def content_pil(self) -> Image:
        return Image.open(fp=self.content_img_path)
    
    @cached_property
    def style_pil(self) -> Image:
        pil: Image = Image.open(fp=self.style_img_path)
        return pil.resize(size=self.content_pil.size)

    @staticmethod
    def __to_tensor(path: os.PathLike, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        tensor: torch.Tensor = torchvision.io.read_image(path=path)
        rgb_mean: torch.Tensor = tensor.mean(dim=[1, 2])    # H, W
        rgb_std: torch.Tensor = tensor.std(dim=[1, 2])      # H, W
        tensor: torch.Tensor = torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)(tensor)
        if target_size is not None:
            tensor: torch.Tensor = torchvision.transforms.Resize(size=target_size)
        return tensor


class Extractor:

    def __init__(
        self, 
        pretrained_net: nn.Module, 
        content_module_names: List[str], 
        style_module_names: List[str]
    ) -> None:
        self._pretrained_net: nn.Module = pretrained_net
        p: torch.nn.parameter.Parameter
        for p in self._pretrained_net.parameters():
            p.requires_grad = False
        self._content_module_names: List[str] = content_module_names
        self._style_module_names: List[str] = style_module_names

    def __call__(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        contents: List[torch.Tensor] = []
        styles: List[torch.Tensor] = []
        name: str
        module: nn.Module
        for name, module in self._pretrained_net.named_modules():
            x: torch.Tensor = module(x)
            if name in self._content_module_names:
                contents.append(x)
            if name in self._style_module_names:
                styles.append(x)
        return {'contents': contents, 'styles': styles}
    

class SynthesizedImage(nn.Module):

    def __init__(self, content_tensor: torch.Tensor):
        super(SynthesizedImage, self).__init__()
        self.weight: nn.Parameter = nn.Parameter(data=content_tensor)

    def forward(self) -> nn.Parameter:
        return self.weight


class Loss:

    def __init__(
        self, 
        content_weight: float = 1., 
        style_weight: float = 1e4, 
        total_variance_weight: float = 10.,
    ) -> None:
        self.content_weight: float = content_weight
        self.style_weight: float = style_weight
        self.total_variance_weight: float = total_variance_weight

    @staticmethod
    def content_loss(gt_content: torch.Tensor, synthesized_content: torch.Tensor) -> torch.Tensor:
        return torch.square(synthesized_content - gt_content.detach()).mean()

    @staticmethod
    def style_loss(gt_style: torch.Tensor, synthesized_style: torch.Tensor) -> torch.Tensor:
        return torch.square(Loss._gram(gt_style) - Loss._gram(synthesized_style)).mean()

    @staticmethod
    def total_variance_loss(x: torch.Tensor) -> torch.Tensor:
        h_diff: torch.Tensor = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        w_diff: torch.Tensor = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return h_diff + w_diff
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        gt_contents: List[torch.Tensor], 
        gt_styles: List[torch.Tensor],
        synthesized_contents: List[torch.Tensor], 
        synthesized_styles: List[torch.Tensor], 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        content_losses: torch.Tensor = torch.stack(
            [
                self.content_loss(gt_content=gt_content, synthesized_content=synthesized_content) * self.content_weight
                for gt_content, synthesized_content in zip(gt_contents, synthesized_contents)
            ],
            dim=0
        )
        style_losses: torch.Tensor = torch.stack(
            [
                self.style_loss(gt_style=gt_style, synthesized_style=synthesized_style) * self.style_weight
                for gt_style, synthesized_style in zip(gt_styles, synthesized_styles)
            ],
            dim=0
        )
        total_variance_loss: torch.Tensor = self.total_variance_loss(x) * self.total_variance_weight
        total_loss: torch.Tensor = torch.sum(torch.stack(content_losses, style_losses, total_variance_loss))
        return content_losses, style_losses, total_variance_loss, total_loss

    def _gram(x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor     # shape = (N, C, H, W)
        batch_size: int = x.shape[0]
        n_channels: int = x.shape[1]
        x: torch.Tensor = x.reshape(batch_size, n_channels, -1)         # (N, C, H * W)
        gram: torch.Tensor = torch.bmm(x, x.transpose(dim0=1, dim1=2))  # (N, C, C)
        return gram / (n_channels * x.shape[-1])    # normalized by C * H * W



if __name__ == '__main__':

    device: torch.device = torch.device('cpu')
    content_img_path: os.PathLike = r'Content_DangVoHiep.jpg'
    style_img_path: os.PathLike = r'Style_LeonardoDaVinci.jpg'
    io: IO = IO(content_img_path=content_img_path, style_img_path=style_img_path)
    synthesized_image: SynthesizedImage = SynthesizedImage(content_tensor=io.content_tensor).to(device)
    trainer: torch.optim.Optimizer = torch.optim.Adam(synthesized_image.parameters(), lr=0.3)
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer=trainer, step_size=500, gamma=0.8)

    extractor = Extractor(
        pretrained_net=torchvision.models.vgg19(pretrained=True),
    )
