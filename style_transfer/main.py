import os
from typing import List
from functools import cached_property
from PIL.Image import Image

import torch
import torchvision
from torch import nn

from utils import Timer, Logger


class IO:

    def __init__(
        self, 
        content_img_path: os.PathLike, 
        style_img_path: os.PathLike, 
        device: torch.device,
    ) -> None:
        self.content_img_path: os.PathLike = content_img_path
        self.style_img_path: os.PathLike = style_img_path
        self.device: torch.device = device
        self.rgb_mean_content: float
        self.rgb_std_content: float
        self.rgb_mean_style: float
        self.rgb_std_style: float

    # Input
    @cached_property
    def content_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = torchvision.io.read_image(path=self.content_img_path)
        tensor: torch.Tensor = tensor.to(dtype=torch.float, device=self.device) / 255.
        self.rgb_mean_content: torch.Tensor = tensor.mean(dim=(1, 2))   # (C, H, W) -> (C,)
        self.rgb_std_content: torch.Tensor = tensor.std(dim=(1, 2))     # (C, H, W) -> (C,)
        tensor: torch.Tensor = (tensor - self.rgb_mean_content.view(-1, 1, 1)) / self.rgb_std_content.view(-1, 1, 1)
        return tensor
    
    # Input
    @cached_property
    def style_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = torchvision.io.read_image(path=self.style_img_path)
        tensor: torch.Tensor = tensor.to(dtype=torch.float, device=self.device) / 255.
        tensor: torch.Tensor = torchvision.transforms.Resize(size=self.content_tensor.shape[-2:])(tensor)
        self.rgb_mean_style: torch.Tensor = tensor.mean(dim=(1, 2))     # (C, H, W) -> (C,)
        self.rgb_std_style: torch.Tensor = tensor.std(dim=(1, 2))       # (C, H, W) -> (C,)
        tensor: torch.Tensor = (tensor - self.rgb_mean_style.view(-1, 1, 1)) / self.rgb_std_style.view(-1, 1, 1)
        return tensor
    
    # Input
    @cached_property
    def content_pil(self) -> Image:
        return Image.open(fp=self.content_img_path)
    
    # Input
    @cached_property
    def style_pil(self) -> Image:
        pil: Image = Image.open(fp=self.style_img_path)
        return pil.resize(size=self.content_pil.size)
    
    # Output
    def save_output(self, x: torch.Tensor, filepath: os.PathLike) -> None:
        tensor: torch.Tensor = x.cpu()
        tensor: torch.Tensor = tensor * self.rgb_std_content.view(-1, 1, 1) + self.rgb_mean_content.view(-1, 1, 1)
        tensor: torch.Tensor = torch.clamp(tensor, min=0., max=1.)
        torchvision.transforms.ToPILImage()(tensor).save(filepath)


class Extractor:

    def __init__(
        self, 
        pretrained_feature_extractor: nn.Module, 
        content_module_names: List[str], 
        style_module_names: List[str]
    ) -> None:
        __first_param: nn.Parameter = next(pretrained_feature_extractor.parameters())
        self.device: torch.device = __first_param.device
        self._pretrained_feature_extractor: nn.Module = pretrained_feature_extractor
        p: torch.nn.parameter.Parameter
        for p in self._pretrained_feature_extractor.parameters():
            p.requires_grad = False
        self._content_module_names: List[str] = content_module_names
        self._style_module_names: List[str] = style_module_names
    
    def extract_contents(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.__extract(x, self._content_module_names)

    def extract_styles(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.__extract(x, self._style_module_names)
    
    def __extract(self, x: torch.Tensor, module_names: List[str]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        name: str
        module: nn.Module
        x: torch.Tensor = x.unsqueeze(dim=0)
        for name, module in self._pretrained_feature_extractor.named_modules():
            if name == '':
                # ignore parent module
                continue
            x: torch.Tensor = module(x)
            if name in module_names:
                results.append(x.squeeze(dim=0))
        return results


class SynthesizedImage(nn.Module):

    def __init__(self, content_tensor: torch.Tensor):
        super(SynthesizedImage, self).__init__()
        self.weight: nn.Parameter = nn.Parameter(data=content_tensor)

    def forward(self) -> nn.Parameter:
        return self.weight


class Loss:

    def __init__(
        self, 
        content_weight: float, 
        style_weight: float, 
        total_variance_weight: float,
    ) -> None:
        self.content_weight: float = content_weight
        self.style_weight: float = style_weight
        self.total_variance_weight: float = total_variance_weight

    @staticmethod
    def content_loss(gt_content: torch.Tensor, synthesized_content: torch.Tensor) -> torch.Tensor:
        return torch.square(synthesized_content - gt_content.detach()).mean()

    @staticmethod
    def style_loss(gt_style: torch.Tensor, synthesized_style: torch.Tensor) -> torch.Tensor:
        return torch.square(Loss.gram(x=gt_style) - Loss.gram(x=synthesized_style)).mean()

    @staticmethod
    def total_variance_loss(x: torch.Tensor) -> torch.Tensor:
        h_diff: torch.Tensor = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
        w_diff: torch.Tensor = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
        return h_diff + w_diff
    
    @staticmethod
    def gram(x: torch.Tensor) -> torch.Tensor:
        n_channels: int = x.shape[0]
        x: torch.Tensor = x.reshape(n_channels, -1) # (C, H * W)
        gram: torch.Tensor = torch.matmul(x, x.transpose(dim0=0, dim1=1))  # (C, C)
        return gram / (n_channels * x.shape[1])    # normalized by C * H * W
    
    def compute_loss(
        self, 
        synthesized_image: SynthesizedImage, 
        gt_contents: List[torch.Tensor], 
        gt_styles: List[torch.Tensor],
        synthesized_contents: List[torch.Tensor], 
        synthesized_styles: List[torch.Tensor], 
    ) -> torch.Tensor:
        content_losses: torch.Tensor = torch.stack(
            tensors=[
                self.content_loss(gt_content=gt_content, synthesized_content=synthesized_content) * self.content_weight
                for gt_content, synthesized_content in zip(gt_contents, synthesized_contents)
            ],
            dim=0,
        )
        style_losses: torch.Tensor = torch.stack(
            tensors=[
                self.style_loss(gt_style=gt_style, synthesized_style=synthesized_style) * self.style_weight
                for gt_style, synthesized_style in zip(gt_styles, synthesized_styles)
            ],
            dim=0,
        )
        total_variance_loss: torch.Tensor = torch.stack(
            tensors=[self.total_variance_loss(x=synthesized_image()) * self.total_variance_weight],
            dim=0,
        )
        total_loss: torch.Tensor = torch.sum(torch.cat(tensors=[content_losses, style_losses, total_variance_loss], dim=0))
        return total_loss


def train(
    content_tensor: torch.Tensor, 
    style_tensor: torch.Tensor,
    extractor: Extractor,
    learning_rate: float = 0.3,
    n_epochs: int = 500,
) -> torch.Tensor:
    
    synthesized_image: nn.Module = SynthesizedImage(content_tensor=content_tensor).to(device=content_tensor.device)
    trainer: torch.optim.Optimizer = torch.optim.Adam(params=synthesized_image.parameters(), lr=learning_rate)
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer=trainer, step_size=25, gamma=0.8)
    loss = Loss(content_weight=10., style_weight=1e4, total_variance_weight=1e-2)

    timer: Timer = Timer()
    logger: Logger = Logger()

    epoch: int
    for epoch in range(1, n_epochs + 1):
        timer.start_epoch(epoch)
        trainer.zero_grad()
        total_loss: torch.Tensor = loss.compute_loss(
            synthesized_image=synthesized_image,
            gt_contents=extractor.extract_contents(x=content_tensor),
            gt_styles=extractor.extract_styles(x=style_tensor),
            synthesized_contents=extractor.extract_contents(x=synthesized_image()),
            synthesized_styles=extractor.extract_styles(x=synthesized_image()),
        )
        total_loss.backward()
        trainer.step()
        scheduler.step()
        timer.end_epoch(epoch)
        logger.log(epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch=epoch), total_loss=total_loss.item())
    
    return synthesized_image.weight.data


if __name__ == '__main__':

    n_epochs: int = 500
    device: torch.device = torch.device('cpu')
    # content_img_path: os.PathLike = r'Content_DangVoHiep.jpg'
    # style_img_path: os.PathLike = r'Style_VincentVanGogh.jpg'
    content_img_path: os.PathLike = r'rainier.jpg'
    style_img_path: os.PathLike = r'autumn-oak.jpg'
    io: IO = IO(content_img_path=content_img_path, style_img_path=style_img_path, device=device)
    extractor: Extractor = Extractor(
        pretrained_feature_extractor=torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT
        ).features.to(device=device),
        content_module_names=['0','5','10','19','28'],
        style_module_names=['25'],
    )
    output_tensor: torch.Tensor = train(
        content_tensor=io.content_tensor, 
        style_tensor=io.style_tensor, 
        extractor=extractor,
        learning_rate=0.3,
        n_epochs=n_epochs, 
    ).detach()
    io.save_output(x=output_tensor, filepath='./output/rainier.jpg')


