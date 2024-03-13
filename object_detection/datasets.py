import os
import typing

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

        super(BananasDataset, self).__init__()
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
    A PyTorch Dataset class for the nuImages dataset that loads images and annotations on-the-fly.
    
    This class is designed to handle large datasets by loading images and their corresponding
    annotations lazily, i.e., as they are needed for training, instead of loading the entire
    dataset into RAM at once. This approach significantly reduces memory consumption and is
    especially useful when working with large datasets.
    
    Attributes:
        - nuim (NuImages): The nuImages dataset object initialized with provided parameters
        - n_categories (int): number of categories
        - n_annotations (int): The maximum number of annotations per image. Defaults to 10
        - category_token_numeric_mapper (dict): A mapping from category tokens to numeric labels
        - category_numeric_name_mapper (dict): A mapping from numeric labels to category name
        - sample_data (list): A list of metadata for keyframe sample images in the dataset.
        - dataroot (str): The root directory where the nuImages data is located
        - version (Literal): The version of the nuImages dataset to use. Can be "v1.0-train", "v1.0-val",
          "v1.0-test", or "v1.0-mini".
        - device (torch.device): The device on which tensors are loaded

    """
    def __init__(
        self, 
        n_annotations: int = 10, 
        dataroot: str = f'{os.environ["PYTHONPATH"]}/data/nuimages', 
        version: typing.Literal["v1.0-train", "v1.0-val", "v1.0-test", "v1.0-mini"] = "v1.0-mini", 
        device: torch.device = torch.device('cpu')
    ):
        
        """
        Parameters:
        - n_annotations (int): The maximum number of annotations per image. Defaults to 10.
        - dataroot (str): The root directory where the nuImages data is located.
        - version (Literal): The version of the nuImages dataset to use. Can be "v1.0-train", "v1.0-val",
          "v1.0-test", or "v1.0-mini".
        - device (torch.device): The device on which tensors are loaded. Defaults to CPU.

        """
        super(NuImagesDataset, self).__init__()
        self.n_annotations = n_annotations
        self.device = device
        self.version = version
        self.dataroot = dataroot
        
        # Initialize the nuImages dataset with the given parameters
        self.nuim = NuImages(dataroot=dataroot, version=version, verbose=True, lazy=True)
        
        # Map categories to numeric labels
        self.category_token_numeric_mapper = {category['token']: float(i) for i, category in enumerate(self.nuim.category)}
        # Map numeric labels to category names
        self.category_numeric_name_mapper = {float(i): category['name'] for i, category in enumerate(self.nuim.category)}

        # Get number of categories
        self.n_categories = len(self.category_token_numeric_mapper)
        
        # Store only the metadata required to load images and annotations on-the-fly
        self.sample_data = [sample for sample in self.nuim.sample_data if sample['is_key_frame']]

    def __getitem__(self, idx: int):
        sample_image: dict = self.sample_data[idx]
        image_token: str = sample_image['token']
        filename: str = sample_image['filename']
        
        # Load image
        image_path = os.path.join(self.dataroot, filename)
        image_tensor = torchvision.io.read_image(path=image_path).to(dtype=torch.float, device=self.device)
        
        # Load annotations
        annotations = []
        count = 0
        for annotation in self.nuim.object_ann:
            if annotation['sample_data_token'] == image_token:
                count += 1
                if count <= self.n_annotations:
                    # Get relavtive bboxes
                    bbox_absolute = annotation['bbox']
                    bbox_relative = [
                        bbox_absolute[0] / sample_image['width'],
                        bbox_absolute[1] / sample_image['height'],
                        bbox_absolute[2] / sample_image['width'],
                        bbox_absolute[3] / sample_image['height'],
                    ]
                    # Get numeric categories
                    category_token = annotation['category_token']
                    category_number = self.category_token_numeric_mapper[category_token]
                    # Append the annotation list
                    annotations.append([category_number] + bbox_relative)
        
        # Pad annotations if needed
        if count < self.n_annotations:
            n_pad = self.n_annotations - count
            annotations.extend([[-1.] + [0.] * 4] * n_pad)
        
        annotations_tensor = torch.tensor(annotations, dtype=torch.float, device=self.device)
        
        return image_tensor, annotations_tensor

    def __len__(self):
        return len(self.sample_data)



if __name__ == '__main__':

    self = NuImagesDataset(n_annotations=10, version='v1.0-train')


