import os
import typing
from collections import defaultdict

import pandas as pd

import torch
import torchvision
import torch.utils.data

from nuimages import NuImages



class BananasDataset(torch.utils.data.Dataset):

    """
    Read the banana detection dataset images and labels.

    Reference: https://d2l.ai/chapter_computer-vision/object-detection-dataset.html
    """

    def __init__(
        self, 
        is_train: bool = True, 
        data_dir: str = f'{os.environ["PYTHONPATH"]}/data/banana-detection',
        device: torch.device = torch.device('cpu'),
    ):
        
        """
        Attributes:
        - n_classes (int): number of classes (= 1)

        Parameters:
        - is_train (bool): Flag indicating whether to load the training or validation dataset.
        - data_dir (str): The root directory of the banana detection dataset.
        - device (torch.device): Where the data tensors should be stored
        """

        # This dataset only has 1 class (banana)
        self.n_classes = 1

        # Define the path to the CSV file containing the labels
        csv_fname = os.path.join(
            data_dir, 
            'bananas_train' if is_train else 'bananas_val', 
            'label.csv'
        )

        # Read the CSV file into a DataFrame and set the image names as the index
        csv_data = pd.read_csv(csv_fname)
        csv_data = csv_data.set_index('img_name')

        features, labels = [], []  # Lists to hold image features and their corresponding labels

        # Iterate over each row in the DataFrame to load images and their targets
        for img_name, target in csv_data.iterrows():
            # Load the image as a tensor
            image_tensor = torchvision.io.read_image(
                os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')
            ).to(dtype=torch.float, device=device)
            features.append(image_tensor)

            # The target contains (class, upper-left x, upper-left y, lower-right x, lower-right y)
            # All images have the same banana class (index 0)
            labels.append(list(target))

        # Convert the list of targets to a tensor and unsqueeze to add an extra dimension 
        # assuming image sizes are consistent
        self.__features: torch.Tensor = torch.stack(features, dim=0)
        
        self.__labels: torch.Tensor = torch.tensor(labels, dtype=torch.float, device=device)
        self.__labels = self.__labels.unsqueeze(dim=1)
        self.__labels[:, :, [1, 3]] /= self.__features.shape[3]
        self.__labels[:, :, [2, 4]] /= self.__features.shape[2]

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)



class NuImagesDataset(torch.utils.data.Dataset):

    """
    A PyTorch Dataset class for loading and using the nuImages dataset.
    """

    def __init__(
        self, 
        n_annotations: int = 10,
        lazy: bool = True,
        dataroot: str = f'{os.environ["PYTHONPATH"]}/data/nuimages',
        version: typing.Literal["v1.0-train", "v1.0-val", "v1.0-test", "v1.0-mini"] = "v1.0-mini",
        device: torch.device = torch.device('cpu'),
    ):
        
        """
        Attributes:
        - nuim (NuImages): The nuImages dataset object.
        - n_categories (int): number of categories

        Parameters:
        - n_annotations (int): The maximum number of annotations per image. Defaults to 10.
        - lazy (bool): If True, images are loaded lazily. Defaults to True.
        - dataroot (str): The root directory where the nuImages data is located.
        - version (Literal): The version of the nuImages dataset to use. Can be "v1.0-train", "v1.0-val", "v1.0-test", or "v1.0-mini".
        - device (torch.device): The device on which tensors are loaded. Defaults to CPU.
        """
        
        # Initialize the nuImages dataset with the given parameters
        self.nuim = NuImages(dataroot=dataroot, version=version, verbose=True, lazy=lazy)
        
        # Map categories to numeric labels
        category_mapper = {category['token']: float(i) for i, category in enumerate(self.nuim.category)}

        # Prepare containers for annotations and image tensors.
        image_annotations = defaultdict(list)
        image_tensors = dict()

        for sample_image in self.nuim.sample_data:
            
            # Only extract keyframes (those have annotations)
            if not sample_image['is_key_frame']:
                continue

            image_token = sample_image['token']
            filename = sample_image['filename']
            
            # Get image tensor
            image_tensors[filename] = torchvision.io.read_image(
                os.path.join(dataroot, filename)
            ).to(dtype=torch.float, device=device)

            # Get annotations
            count = 0
            for annotation in self.nuim.object_ann:
                if annotation['sample_data_token'] == image_token:
                    count += 1
                    if count <= n_annotations:
                        # Get relavtive bboxes
                        width = sample_image['width']
                        height = sample_image['height']
                        bbox_absolute = annotation['bbox']
                        bbox_relavtive = [
                            bbox_absolute[0] / width,
                            bbox_absolute[1] / height,
                            bbox_absolute[2] / width,
                            bbox_absolute[3] / height,
                        ]
                        # Get numeric categories
                        category_token = annotation['category_token']
                        category_number = category_mapper[category_token]
                        image_annotations[filename].append([category_number] + bbox_relavtive)

            # Pad annotations if there are less than `n_annotations`.
            if count < n_annotations:
                n_pad = n_annotations - count
                image_annotations[filename].extend([[-1.] + [0.] * 4] * n_pad)

        # Convert annotations to tensors.
        image_annotations = {
            filename: torch.tensor(bboxes, dtype=torch.float, device=device) 
            for filename, bboxes in image_annotations.items()
        }

        # Prepare the dataset records as a list of (image_tensor, annotations) tuples.
        self.__records = [
            (image_tensors[filename], image_annotations[filename])
            for filename in image_tensors.keys()
        ]
        self.n_categories = len(category_mapper.keys())


    def __getitem__(self, idx: int):
        return self.__records[idx]

    def __len__(self):
        return len(self.__records)


if __name__ == '__main__':

    self = NuImagesDataset()


