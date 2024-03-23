import os
import typing
from PIL import Image

import numpy as np

import torch
import torchvision
import torchvision.transforms
import torchvision.transforms.functional as F
import torch.utils.data


class VOC2012(torch.utils.data.Dataset):

    """
    Dataset class for the VOC2012 dataset.
    """

    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]

    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
    ]

    def __init__(
        self,
        datadir: str = f'{os.environ["PYTHONPATH"]}/data/voc2012',
        is_train: bool = True,
        image_size: typing.Tuple[int, int] = (320, 480),
        device: torch.device = torch.device('cpu'),
    ) -> None:
        """
        Initialize the VOC2012 dataset.

        Args:
        - datadir: The directory where the VOC2012 dataset is located.
        - is_train: Whether to use the training set or the validation set.
        - device: The device to load the tensors on.
        """
        super(VOC2012, self).__init__()
        self.is_train = is_train
        self.datadir = datadir
        self.image_size = image_size
        self.device = device
        filename_txt = os.path.join(datadir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
        with open(filename_txt, mode='r') as file:
            self._filenames = [line.replace('\n', '') for line in file.readlines()]

    def __getitem__(self, idx) -> typing.Tuple:
        """
        Get the image and label at the given index.

        Args:
        - idx: The index to get the image and label from.
            
        Returns:
        - A tuple of the image and label tensors.
        """

        # Read the image file
        image_filename = os.path.join(self.datadir, 'JPEGImages', self._filenames[idx] + '.jpg')
        image_tensor = torchvision.io.read_image(
            path=image_filename, 
            mode=torchvision.io.image.ImageReadMode.RGB
        ).to(dtype=torch.float, device=self.device)
        image_tensor /= 255.

        # Read the corresponding label file
        label_filename = os.path.join(self.datadir, 'SegmentationClass', self._filenames[idx] + '.png')
        image_label_3d = torchvision.io.read_image(
            path=label_filename, 
            mode=torchvision.io.image.ImageReadMode.RGB
        ).to(dtype=torch.uint8, device=self.device)
        image_label_2d = self.__convert_label_3d_to_2d(image_label_3d)
        
        # Random crop the image to the common size and return
        return self.__random_crop(image_tensor, image_label_2d, size=self.image_size)
        
    def __len__(self):
        """
        Get the number of images in the dataset.

        Returns:
        - The number of images in the dataset.
        """
        return len(self._filenames)

    @classmethod
    def __convert_label_3d_to_2d(cls, image_label_3d: torch.Tensor) -> torch.Tensor:
        """
        Convert a 3D label tensor to a 2D label tensor.

        Args:
        - image_label_3d: The 3D label tensor to convert.

        Returns:
        - The converted 2D label tensor.
        """
        # Convert the 3D tensor to a 2D tensor where each element is a class index
        image_label_2d = torch.zeros(size=image_label_3d.shape[1:], dtype=torch.long, device=image_label_3d.device)
        # Convert the 3D tensor to a 2D tensor where each element is an RGB value
        image_label_rgb = image_label_3d.permute(1, 2, 0).reshape(-1, 3)
        # Convert the RGB values to class indices
        for i, rgb in enumerate(cls.VOC_COLORMAP):
            mask = (image_label_rgb == torch.tensor(rgb, dtype=torch.uint8, device=image_label_3d.device)).all(dim=1)
            image_label_2d[mask.reshape(*image_label_2d.shape)] = i

        return image_label_2d

    @classmethod
    def __random_crop(
        cls,
        image_tensor: torch.Tensor, 
        image_label: torch.Tensor, 
        size: typing.Tuple[int, int]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Apply a random crop to the image and label tensors.

        Args:
        - image_tensor: The image tensor to crop.
        - image_label: The label tensor to crop (either 2d or 3d).
        - size: The desired output size of the crop.
        
        Returns:
            A tuple of the cropped image and label tensors.
        """
        image_tensor = cls.__image_pad(image_tensor=image_tensor, min_size=size)

        random_rect: typing.Tuple[int, int, int, int] = torchvision.transforms.RandomCrop.get_params(
            img=image_tensor,
            output_size=size,
        )
        cropped_image_tensor = F.crop(image_tensor, *random_rect)
        cropped_label_tensor = F.crop(image_label, *random_rect)
        return cropped_image_tensor, cropped_label_tensor

    @staticmethod
    def __image_pad(
        image_tensor: torch.Tensor,
        min_size: typing.Tuple[int, int]
    ) -> torch.Tensor:
        """
        Pad an image tensor to a specified size.

        If the image tensor's size is smaller than the specified size, this function pads the edges of the image tensor with the edge values.
        If the image tensor's size is larger than or equal to the specified size, returns the original image tensor.

        Args:
        - image_tensor (torch.Tensor): The image tensor to pad. It should have shape (C, H, W).
        - min_size (typing.Tuple[int, int]): The minimum size (height, width) of the output image.

        Returns:
        - torch.Tensor: The padded image tensor. It has the same number of channels as the input image tensor, and its height and width 
        are at least as large as the specified size.
        """
        # Get the current size of the image tensor
        _, h, w = F.get_dimensions(image_tensor)
        # Get the target size
        th, tw = min_size

        # Calculate the amount of padding needed for each side
        need_h = max(th - h, 0)
        need_w = max(tw - w, 0)

        # Distribute the padding evenly to the top/bottom and left/right sides
        pad_top = need_h // 2
        pad_bottom = need_h - pad_top
        pad_left = need_w // 2
        pad_right = need_w - pad_left

        # Pad the image tensor and return it
        return F.pad(
            img=image_tensor, 
            padding=(pad_left, pad_top, pad_right, pad_bottom),    #  left, top, right and bottom
            padding_mode='edge'
        )
    
    def show_sample(self, idx: int, filepath: os.PathLike) -> None:
        """
        This method save to a file a sample image and its corresponding label.

        Parameters:
        - idx (int): The index of the sample to display.
        - filepath (PathLike): The path where the resulting image will be saved.

        Returns:
        - None
        """
        # Get image tensor and label tensor
        image_tensor: torch.Tensor
        label_tensor: torch.Tensor
        image_tensor, label_tensor = self[idx]
        # Concatenate the image and label arrays along the horizontal axis
        image_np = (image_tensor * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()  # H, W, C
        label_np = np.array(self.VOC_COLORMAP, dtype=np.uint8)[label_tensor.cpu().numpy()]
        # Save the resulting image to the specified filepath
        Image.fromarray(np.concatenate([image_np, label_np], axis=1)).save(filepath)


if __name__ == '__main__':
    self = VOC2012()
    for i in range(50):
        self.show_sample(i, f'/Users/hiepdang/vision/sample{i}.jpg')


