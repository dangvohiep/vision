from typing import List, Tuple

import torch
from torch import nn

from .. import generate_anchors


class SingleShotDetection(nn.Module):
    """
    TinySSD class is a simplified Single Shot MultiBox Detector model for object detection.

    Parameters:
    - n_classes (int): Number of classes in the dataset, excluding the background class.
    """

    def __init__(
        self,
        n_classes: int,
        sizes: List[List[float]] = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
        ratios: List[List[float]] = [[1., 2., 0.5]] * 5
    ):
        super(SingleShotDetection, self).__init__()
        self.sizes: List[List[float]] = sizes
        self.ratios: List[List[float]] = ratios
        self.n_anchors: int = len(sizes[0]) + len(ratios[0]) - 1   # number of anchors centered at each pixel
        self.n_classes: int = n_classes

        # Base network layers (Block 0)
        self.feature_extractor_0: nn.Sequential = nn.Sequential(
            # Down-sample block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Down-sample block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Down-sample block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.class_predictor_0: nn.Conv2d = nn.Conv2d(
            in_channels=64, 
            out_channels=self.n_anchors * (n_classes + 1), 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bbox_predictor_0: nn.Conv2d = nn.Conv2d(
            in_channels=64, 
            out_channels=self.n_anchors * 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        # Block 1
        self.feature_extractor_1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.class_predictor_1: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * (n_classes + 1), 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bbox_predictor_1: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        # Block 2
        self.feature_extractor_2: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.class_predictor_2: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * (n_classes + 1), 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bbox_predictor_2: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * 4, 
            kernel_size=3, 
            stride=1,
            padding=1,
        )
        # Block 3
        self.feature_extractor_3: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.class_predictor_3: nn.Conv2d = nn.Conv2d(
            in_channels=128,
            out_channels=self.n_anchors * (n_classes + 1),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bbox_predictor_3: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        # Block 4
        self.adapt_pool: nn.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.class_predictor_4: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * (n_classes + 1), 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bbox_predictor_4: nn.Conv2d = nn.Conv2d(
            in_channels=128, 
            out_channels=self.n_anchors * 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Block 0
        x: torch.Tensor = self.feature_extractor_0(x)                   # (32, 3, 256, 256) -> (32, 64, 32, 32)
        anchor_0: torch.Tensor = generate_anchors(data=x, sizes=self.sizes[0], ratios=self.ratios[0]) # (32, 4096, 4)
        class_prediction_0: torch.Tensor = self.class_predictor_0(x)    # (32, 8, 32, 32)
        bbox_prediction_0: torch.Tensor = self.bbox_predictor_0(x)      # (32, 16, 32, 32)

        # Block 1
        x: torch.Tensor = self.feature_extractor_1(x)                   # (32, 128, 16, 16)
        anchor_1: torch.Tensor = generate_anchors(data=x, sizes=self.sizes[1], ratios=self.ratios[1]) # (32, 1024, 4)
        class_prediction_1: torch.Tensor = self.class_predictor_1(x)    # (32, 8, 16, 16)
        bbox_prediction_1: torch.Tensor = self.bbox_predictor_1(x)      # (32, 16, 16, 16)

        # Block 2
        x: torch.Tensor = self.feature_extractor_2(x)                   # (32, 128, 8, 8)
        anchor_2: torch.Tensor = generate_anchors(data=x, sizes=self.sizes[2], ratios=self.ratios[2]) # (32, 256, 4)
        class_prediction_2: torch.Tensor = self.class_predictor_2(x)    # (32, 8, 8, 8)
        bbox_prediction_2: torch.Tensor = self.bbox_predictor_2(x)      # (32, 16, 8, 8)

        # Block 3
        x: torch.Tensor = self.feature_extractor_3(x)                   # (32, 128, 4, 4)
        anchor_3: torch.Tensor = generate_anchors(data=x, sizes=self.sizes[3], ratios=self.ratios[3]) # (32, 64, 4)
        class_prediction_3: torch.Tensor = self.class_predictor_3(x)    # (32, 8, 4, 4)
        bbox_prediction_3: torch.Tensor = self.bbox_predictor_3(x)      # (32, 16, 4, 4)

        # Block 4
        x: torch.Tensor = self.adapt_pool(x)                            # (32, 128, 1, 1)
        anchor_4: torch.Tensor = generate_anchors(data=x, sizes=self.sizes[4], ratios=self.ratios[4]) # (32, 4, 4)
        class_prediction_4: torch.Tensor = self.class_predictor_4(x)    # (32, 8, 1, 1)
        bbox_prediction_4: torch.Tensor = self.bbox_predictor_4(x)      # (32, 16, 1, 1)

        anchors: List[torch.Tensor] = [
            anchor_0, anchor_1, anchor_2, anchor_3, anchor_4
        ]
        class_predictions: List[torch.Tensor] = [
            class_prediction_0, class_prediction_1, class_prediction_2, class_prediction_3, class_prediction_4
        ]
        bbox_predictions: List[torch.Tensor] = [
            bbox_prediction_0, bbox_prediction_1, bbox_prediction_2, bbox_prediction_3, bbox_prediction_4
        ]

        anchors: torch.Tensor = torch.cat(tensors=anchors, dim=1)       # (32, 5444, 4)
        class_predictions: torch.Tensor = torch.cat(
            [torch.flatten(p.permute(0, 2, 3, 1), start_dim=1) for p in class_predictions], 
            dim=1
        ) # (32, 10888)
        class_predictions: torch.Tensor = class_predictions.reshape(
            class_predictions.shape[0], -1, self.n_classes + 1
        ) # (32, 5444, 2)
        bbox_predictions: torch.Tensor = torch.cat(
            [torch.flatten(p.permute(0, 2, 3, 1), start_dim=1) for p in bbox_predictions], 
            dim=1
        ) # (32, 21776)
        bbox_predictions: torch.Tensor = bbox_predictions.reshape(bbox_predictions.shape[0], -1, 4) # (32, 5444, 4)

        return anchors, class_predictions, bbox_predictions

