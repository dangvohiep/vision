import torch
import torchvision
import torch.nn as nn


from typing import Tuple

class _DownSamplingBlock(nn.Module):
    """
    A class used to represent a down-sampling block in a Fully Convolutional Network (FCN).

    This block uses a pre-trained ResNet18 model (without the last two layers) for down-sampling.
    The parameters of the ResNet18 model are frozen, i.e., they are not updated during training.
    """

    def __init__(self):
        """
        Initialize the down-sampling block.
        """
        super(_DownSamplingBlock, self).__init__()

        # Load the pre-trained ResNet18 model
        pretrained_resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Get all layers except the last two
        self.__named_extractors = list(pretrained_resnet18.named_children())[:-2]
        # Add the layers to this module
        for name, module in self.__named_extractors:
            setattr(self, name, module)
        # Freeze the parameters of the layers
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the down-sampling block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Pass the input through each layer in order
        for module in self.children():
            x = module(x)
        return x
    
    def __getitem__(self, idx: int) -> Tuple[str, nn.Module]:
        """
        Get a layer by its index.

        Args:
            idx (int): The index of the layer.

        Returns:
            Tuple[str, nn.Module]: The name and the module of the layer.
        """
        return self.__named_extractors[idx]
    
    def __len__(self) -> int:
        """
        Get the number of layers in the block.

        Returns:
            int: The number of layers.
        """
        return len(self.__named_extractors)


class _UpSamplingBlock(nn.Module):
    """
    A class used to represent an up-sampling block in a Fully Convolutional Network (FCN).

    This block consists of five sub-blocks, each of which includes a transposed convolution layer,
    a batch normalization layer, and a ReLU activation layer (except for the last sub-block, which
    does not have the ReLU layer).
    """

    def __init__(self, out_channels: int) -> None:
        """
        Initialize the up-sampling block.

        Args:
            out_channels (int): The number of output channels for the transposed convolution layers.
        """
        super(_UpSamplingBlock, self).__init__()
        self.out_channels = out_channels
        self._blocks = nn.Sequential()

        # Create the sub-blocks
        for i in range(5):
            block = nn.Sequential()
            # Add a transposed convolution layer
            block.add_module(f'tconv{i}', nn.LazyConvTranspose2d(out_channels=out_channels, kernel_size=2, stride=2))
            # Add a batch normalization layer
            block.add_module(f'bn{i}', nn.LazyBatchNorm2d())
            # Add a ReLU activation layer (except for the last sub-block)
            if i != 4:  # No ReLU for last block
                block.add_module(f'relu{i}', nn.ReLU(inplace=True))
            # Add the sub-block to the main block
            self._blocks.add_module(f'block{i}', block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the up-sampling block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self._blocks(x)
    
    def __getitem__(self, idx: int) -> nn.Module:
        """
        Get a sub-block by its index.

        Args:
            idx (int): The index of the sub-block.

        Returns:
            nn.Module: The sub-block.
        """
        return self._blocks[idx]
            
    def __len__(self) -> int:
        """
        Get the number of sub-blocks in the block.

        Returns:
            int: The number of sub-blocks.
        """
        return len(self._blocks)


class FullyConvolutionalNetwork(nn.Module):
    """
    A class used to represent a Fully Convolutional Network (FCN).

    This network consists of a down-sampling block and an up-sampling block.
    The down-sampling block uses a pre-trained ResNet18 model for feature extraction,
    and the up-sampling block uses transposed convolution layers for up-sampling.
    """

    def __init__(self, out_channels: int) -> None:
        """
        Initialize the FCN.

        Args:
            out_channels (int): The number of output channels for the up-sampling block.
        """
        super(FullyConvolutionalNetwork, self).__init__()
        self.out_channels = out_channels
        # Create the down-sampling block
        self.down_sampling_block = _DownSamplingBlock()
        # Create the up-sampling block
        self.up_sampling_block = _UpSamplingBlock(out_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Pass the input through the down-sampling block
        x = self.down_sampling_block(x)
        # Pass the result through the up-sampling block
        x = self.up_sampling_block(x)
        return x


if __name__ == '__main__':
    n_classes = 10
    net = FullyConvolutionalNetwork(out_channels=n_classes)
    x = torch.randn(size=(32, 3, 320, 480))
    y = net(x)
    print(y.shape)





