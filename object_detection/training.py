import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F

from utils import Accumulator
import object_detection


def loss_function(
    class_predictions: torch.Tensor,
    class_labels: torch.Tensor,
    bbox_predictions: torch.Tensor,
    bbox_labels: torch.Tensor,
    bbox_masks: torch.Tensor,
) -> torch.Tensor:
    
    """
    Calculate the loss for class predictions and bounding box predictions.

    Parameters:
    - class_predictions (torch.Tensor): The class predictions tensor with shape
      (batch_size, n_anchors, n_classes) where n_classes includes the background class.
    - class_labels (torch.Tensor): The class labels tensor with shape
      (batch_size, n_anchors).
    - bbox_predictions (torch.Tensor): The bounding box predictions tensor with shape
      (batch_size, n_anchors, 4), representing the predicted offsets for each anchor.
    - bbox_labels (torch.Tensor): The ground truth bounding box labels with shape
      (batch_size, n_anchors, 4), representing the true offsets for each anchor.
    - bbox_masks (torch.Tensor): A mask tensor with shape (batch_size, n_anchors, 4)
      used to filter out the padding bounding boxes in the loss calculation.

    Returns:
    - torch.Tensor: A tensor representing the total loss (classification loss + bounding box loss)
      averaged over the batch.

    The function computes the classification loss using cross entropy and the bounding box loss using
    Smooth L1 loss. It then combines these two losses into a single total loss value per example.

    """
    # Flatten the predictions and labels to compute the loss across all anchors
    batch_size, n_classes = class_predictions.shape[0], class_predictions.shape[2]
    class_prediction_loss = torch.nn.functional.cross_entropy(
        input=class_predictions.reshape(-1, n_classes), # shape = (174208, 2)
        target=class_labels.reshape(-1),                # shape = (174208,)
        reduction='none',
    )                                                   # shape = (174208,)
    class_prediction_loss = class_prediction_loss.reshape(batch_size, -1).mean(dim=1)   # (32,) --> avg across all anchors

    # Apply the bounding box mask before calculating the Smooth L1 loss
    bbox_prediction_loss = torch.nn.functional.smooth_l1_loss(
        input=bbox_predictions * bbox_masks,            # (32, 5444, 4)
        target=bbox_labels * bbox_masks,                # (32, 5444, 4)
        reduction='none',
    )                                                   # (32, 5444, 4)
    bbox_prediction_loss = bbox_prediction_loss.mean(dim=(1, 2))  # (32,) --> avg across all bboxes
    # Combine the classification and bounding box losses
    alpha = 0.5
    return alpha * class_prediction_loss + (1 - alpha) * bbox_prediction_loss # (32,)


def classification_eval(class_predictions: torch.Tensor, class_labels: torch.Tensor) -> float:
    """
    Evaluate the classification accuracy by comparing predictions with labels.

    Parameters:
    - class_predictions (torch.Tensor): The tensor containing the class predictions. This tensor should
      have shape (batch_size, n_classes) where `n_classes` represents the number of possible
      classes, and each element is the predicted score for each class.
    - class_labels (torch.Tensor): The tensor containing the true class labels. This tensor should
      have the same batch_size as `class_predictions` and contains the actual class labels.

    Returns:
    - float: The total number of correct predictions converted to a float.

    The function computes the accuracy by finding the class with the highest score (using `argmax`)
    for each prediction, then compares these predicted classes with the true labels to count the
    number of correct predictions.
    """
    predicted_classes = class_predictions.argmax(dim=-1).type(class_labels.dtype)
    correct_predictions = (predicted_classes == class_labels).sum()

    return float(correct_predictions)


def bbox_eval(bbox_predictions: torch.Tensor, bbox_labels: torch.Tensor, bbox_masks: torch.Tensor) -> float:
    """
    Evaluate the bounding box predictions using L1 loss, taking into account specified masks.

    Parameters:
    - bbox_predictions (torch.Tensor): The tensor containing the bounding box predictions. This tensor
      should have the shape of (batch_size, n_anchors, 4), where each prediction consists of four
      values (e.g., offset coordinates).
    - bbox_labels (torch.Tensor): The tensor containing the true bounding box labels with the same
      shape as `bbox_predictions`.
    - bbox_masks (torch.Tensor): A tensor of the same shape as `bbox_predictions` and `bbox_labels` used
      to mask certain bounding boxes during the evaluation. This can be used to ignore padding
      bounding boxes or those not relevant to the loss calculation.

    Returns:
    - float: The sum of the absolute differences (L1 loss) between the predicted and true bounding
      boxes, after applying the specified masks, converted to a float.

    The function computes the L1 loss for each predicted bounding box relative to the true labels,
    applies the mask to ignore certain boxes (e.g., padding boxes), and then sums the result to
    provide a single scalar value representing the overall deviation of the predictions from the
    true labels.
    """
    # Calculate the absolute differences between the predicted and true bounding boxes
    # Multiply by `bbox_masks` to zero out the contributions of masked bounding boxes
    l1_loss = torch.abs((bbox_labels - bbox_predictions) * bbox_masks)

    return l1_loss.sum().item()


def train(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    n_epochs: int,
):

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    metrics = Accumulator()

    # loop through each epoch
    for epoch in range(n_epochs):
        # Loop through each batch
        for batch, (batch_images, gt_labels) in enumerate(train_dataloader):    # (N, 3, 256, 256), (N, 1, 5)
            # Reset gradients
            optimizer.zero_grad()
            # Generate multiscale anchor boxes and predict their classes and offsets
            anchors, pred_classes, pred_offsets = model(batch_images)
            # Label the classes and offsets of these anchor boxes
            gt_offsets, gt_bbox_masks, gt_classes = object_detection.compute_groundtruth(
                anchors=anchors,
                labels=gt_labels,
            )
            # Calculate the loss function using the predicted and labeled values of the classes and offsets
            loss = loss_function(
                class_predictions=pred_classes,
                class_labels=gt_classes,
                bbox_predictions=pred_offsets,
                bbox_labels=gt_offsets,
                bbox_masks=gt_bbox_masks,
            ).mean()
            loss.backward()
            optimizer.step()

            # Accumulate the metrics
            metrics.add(
                correct_predictions=classification_eval(class_predictions=pred_classes, class_labels=gt_classes),
                n_predictions=gt_classes.numel(),
                bbox_mae=bbox_eval(bbox_predictions=pred_offsets, bbox_labels=gt_offsets, bbox_masks=gt_bbox_masks),
                n_mae=gt_offsets.numel(),
                loss=loss,
            )
            classification_error = 1 - metrics['correct_predictions'] / metrics['n_predictions']
            bbox_mae = metrics['bbox_mae'] / metrics['n_mae']
            avg_loss = metrics['loss'] / (batch + 1)
            print(
                f'Epoch {epoch + 1}/{n_epochs} || '
                f'Batch {batch + 1}/{len(dataset) // batch_size + 1}: '
                f'classification error: {classification_error:.2e}, '
                f'bbox MAE: {bbox_mae:.2e}, '
                f'loss: {avg_loss:.2e}'
            )
        metrics.reset()
        print('='*20)
    return model


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:

    """
    Perform object detection prediction on an input batch of images.

    This function applies a trained model to the input images to obtain predictions.
    It then processes these predictions to filter and refine them based on the model's
    confidence scores and the predicted bounding box offsets.

    Parameters:
    - model (nn.Module): The trained object detection model.
    - X (torch.Tensor): A batch of input images. The tensor is expected to have the shape
      (batch_size, C, H, W), where C, H, W correspond to the number of channels, height, 
      and width of the images, respectively.

    Returns:
    - torch.Tensor: A tensor containing filtered predictions for each image in the batch.
      The shape of the tensor is (batch_size, n_anchors, 6), where each
      prediction consists of [class_label, score, x_min, y_min, x_max, y_max]
      for each anchor.
    """

    model.eval()
    anchors, pred_classes, pred_offsets = model(X)
    pred_probs = F.softmax(pred_classes, dim=2)

    return object_detection.filter_predictions(
        cls_probs=pred_probs,
        pred_offsets=pred_offsets,
        anchors=anchors,
    )
