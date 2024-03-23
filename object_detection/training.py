import typing

import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F

from utils import Accumulator, EarlyStopping
from object_detection import compute_groundtruth, filter_predictions


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
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    train_batch_size: int,
    val_batch_size: int,
    n_epochs: int,
    patience: int,
    tolerance: float,
    checkpoint_output: typing.Optional[str] = None,
) -> nn.Module:
    """
    Parameters:
    - model (nn.Module): The neural network model to train.
    - train_dataset (torch.utils.data.Dataset): The dataset to train the model on.
    - val_dataset (torch.utils.data.Dataset): The dataset to evaluate the model on.
    - optimizer (torch.optim.Optimizer): The optimizer to use for training.
    - train_batch_size (int): The size of each training batch.
    - val_batch_size (int): The size of each evaluation batch.
    - n_epochs (int): The number of epochs to train the model for.
    - patience (int): Number of epochs with no improvement after which training will be stopped.
    - tolerance (float): The minimum change in eval_loss. Defaults to 0
    - checkpoint_output (Optional[str]): The directory path to save model checkpoints after each epoch. 
      If None, checkpoints are not saved.

    Returns:
    - (nn.Module) The trained model.

    This function directly modifies the model passed to it by updating its weights based on the
    computed gradients during the training process. It also optionally saves the model's state
    at the end of each epoch if a checkpoint output directory is provided.
    """

    model.train()
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    train_metrics = Accumulator()
    early_stopping = EarlyStopping(patience, tolerance)

    # loop through each epoch
    for epoch in range(n_epochs):
        # Loop through each batch
        for batch, (batch_images, gt_labels) in enumerate(train_dataloader):    # (N, 3, 256, 256), (N, 1, 5)
            # Reset gradients
            optimizer.zero_grad()
            # Generate multiscale anchor boxes and predict their classes and offsets
            anchors, pred_classes, pred_offsets = model(batch_images)
            # Label the classes and offsets of these anchor boxes
            gt_offsets, gt_bbox_masks, gt_classes = compute_groundtruth(
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
            train_metrics.add(
                correct_predictions=classification_eval(class_predictions=pred_classes, class_labels=gt_classes),
                n_predictions=gt_classes.numel(),
                bbox_mae=bbox_eval(bbox_predictions=pred_offsets, bbox_labels=gt_offsets, bbox_masks=gt_bbox_masks),
                n_mae=gt_offsets.numel(),
                loss=loss,
            )
            train_classification_error = 1 - train_metrics['correct_predictions'] / train_metrics['n_predictions']
            train_bbox_mae = train_metrics['bbox_mae'] / train_metrics['n_mae']
            train_loss = train_metrics['loss'] / (batch + 1)
            print(
                f'Epoch {epoch + 1}/{n_epochs} || '
                f'Batch {batch + 1}/{len(train_dataloader)} || '
                f'train_classification_error: {train_classification_error:.2e}, '
                f'train_bbox_mae: {train_bbox_mae:.2e}, '
                f'train_loss: {train_loss:.2e}'
            )
        
        # Save checkpoint
        if checkpoint_output:
            torch.save(model, f'{checkpoint_output}/epoch{epoch + 1}.pt')

        # Reset metric records for next epoch
        train_metrics.reset()

        val_classification_error, val_bbox_mae, val_loss = evaluate(model=model, dataset=val_dataset, batch_size=val_batch_size)
        print(
            f'Epoch {epoch + 1}/{n_epochs} || '
            f'val_classification_error: {val_classification_error:.2e}, '
            f'val_bbox_mae: {val_bbox_mae:.2e}, '
            f'val_loss: {val_loss:.2e}'
        )
        print('='*20)

        early_stopping(val_loss)
        if early_stopping:
            print('Early Stopped')
            break
    
    return model


def evaluate(
    model: nn.Module, 
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> typing.Tuple[float, float, float]:
    """
    Evaluate an object detection model on a given dataset.

    Parameters:
    - model (nn.Module): The object detection model to be evaluated.
    - dataset (torch.utils.data.Dataset): The dataset on which the model is evaluated.
    - batch_size (int): size of each evaluation batch.

    Returns:
    - Tuple[float, float, torch.Tensor]: Returns a tuple containing the classification error,
      the mean absolute error (MAE) for bounding box predictions, and the mean loss across all samples.
    """

    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    metrics = Accumulator()

    # Loop through each batch
    for batch, (batch_images, gt_labels) in enumerate(dataloader):    # (N, 3, 256, 256), (N, 1, 5)
        # Predict the anchor locations, class probabilities, and bounding box offsets from the model.
        anchors, pred_classes, pred_offsets = model(batch_images)
        # Compute the ground truth offsets, masks for bounding boxes, and class labels for evaluation.
        gt_offsets, gt_bbox_masks, gt_classes = compute_groundtruth(
            anchors=anchors,
            labels=gt_labels,
        )
        # Calculate the loss using predictions and ground truth values.
        loss = loss_function(
            class_predictions=pred_classes,
            class_labels=gt_classes,
            bbox_predictions=pred_offsets,
            bbox_labels=gt_offsets,
            bbox_masks=gt_bbox_masks,
        ).mean().item()  # Mean loss over the evaluation dataset.
        
        # Accumulate the metrics
        metrics.add(
            correct_predictions=classification_eval(class_predictions=pred_classes, class_labels=gt_classes),
            n_predictions=gt_classes.numel(),
            bbox_mae=bbox_eval(bbox_predictions=pred_offsets, bbox_labels=gt_offsets, bbox_masks=gt_bbox_masks),
            n_mae=gt_offsets.numel(),
            loss=loss,
        )
    
    # Compute the aggregate metrics
    classification_error = 1 - metrics['correct_predictions'] / metrics['n_predictions']
    bbox_mae = metrics['bbox_mae'] / metrics['n_mae']
    loss = metrics['loss'] / (batch + 1)
        
    return classification_error, bbox_mae, loss


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

    return filter_predictions(cls_probs=pred_probs, pred_offsets=pred_offsets, anchors=anchors)

