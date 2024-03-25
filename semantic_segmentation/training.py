import os
import datetime as dt
import typing

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Accumulator, EarlyStopping, Timer, Logger, CheckPointSaver


def loss_function(
    pred_probs: torch.Tensor,
    gt_labels: torch.Tensor,
):
    loss_matrix = F.cross_entropy(input=pred_probs, target=gt_labels, reduction='none')
    return loss_matrix.mean(dim=(1, 2))


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
    checkpoint_dir: typing.Optional[str] = None,
) -> nn.Module:

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    train_metrics = Accumulator()
    early_stopping = EarlyStopping(patience, tolerance)
    timer = Timer()
    logger = Logger()
    checkpoint_saver = CheckPointSaver(dirpath=checkpoint_dir)

    # loop through each epoch
    for epoch in range(1, n_epochs + 1):
        timer.start_epoch(epoch)
        # Loop through each batch
        for batch, (batch_images, gt_labels) in enumerate(train_dataloader, start=1):    # (N, 3, 256, 256), (N, 1, 5)
            timer.start_batch(epoch, batch)
            optimizer.zero_grad()
            pred_probs: torch.Tensor = model(batch_images)
            pred_labels = pred_probs.max(dim=1).indices
            n_corrects = (pred_labels == gt_labels).sum().item()
            n_predictions = pred_labels.numel()
            loss = loss_function(pred_probs, gt_labels).mean()
            loss.backward()
            optimizer.step()
            
            # Accumulate the metrics
            train_metrics.add(n_correct=n_corrects, n_predictions=n_predictions, loss=loss.item())
            train_accuracy = train_metrics['n_correct'] / train_metrics['n_predictions']
            train_loss = train_metrics['loss'] / batch
            timer.end_batch(epoch=epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, batch=batch, n_batches=len(train_dataloader), took=timer.time_batch(epoch, batch),
                train_accuracy=train_accuracy, train_loss=train_loss
            )

        # Save checkpoint
        if checkpoint_dir:
            checkpoint_saver.save(model, filename=f'epoch{epoch}.pt')
        # Reset metric records for next epoch
        train_metrics.reset()
        # Evaluate
        val_accuracy, val_loss = evaluate(model=model, dataset=val_dataset, batch_size=val_batch_size)
        timer.end_epoch(epoch)
        logger.log(epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), val_accuracy=val_accuracy, val_loss=val_loss)
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

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    metrics = Accumulator()

    # Loop through each batch
    for batch, (batch_images, gt_labels) in enumerate(dataloader):    # (N, 3, 256, 256), (N, 1, 5)
        pred_probs: torch.Tensor = model(batch_images)
        pred_labels = pred_probs.max(dim=1).indices
        n_corrects = (pred_labels == gt_labels).sum().item()
        n_predictions = pred_labels.numel()
        loss = loss_function(pred_probs, gt_labels).mean()

        # Accumulate the metrics
        metrics.add(n_corrects=n_corrects, n_predictions=n_predictions, loss=loss.item())

    # Compute the aggregate metrics
    accuracy = metrics['n_corrects'] / metrics['n_predictions']
    loss = metrics['loss'] / (batch + 1)
    return accuracy, loss


def predict(
    model: nn.Module, 
    X: torch.Tensor, 
    colormap: typing.List[typing.Tuple[int, int, int]], 
    save_dir: str,
) -> None:

    # FIXME: create save_dir if not exists
    model.eval()
    pred_probs: np.ndarray = model(X).detach().cpu().numpy()
    pred_labels = pred_probs.argmax(axis=1) # Get the predicted labels

    # Convert the tensor to a numpy array and transpose it to (H, W, C) format
    X_np = X.permute(0, 2, 3, 1).detach().cpu().numpy()

    # loop over each image in the batch
    for i in range(X_np.shape[0]):
        X_i = X_np[i]
        pred_prob_i = pred_probs[i]
        pred_label_i = pred_labels[i]
        # Normalize the original image to the range [0, 255]
        X_i = (X_i * 255).astype(np.uint8)
        # Map the predicted labels to the VOC colormap
        pred_color_label = np.array(colormap)[pred_label_i].astype(np.uint8)
        # Concatenate the original image and the predicted labels along the channel dimension
        result_np = np.concatenate((X_i, pred_color_label), axis=1)
        # Convert the numpy array to a PIL image
        result_pil = Image.fromarray(result_np)
        # Save the image
        result_pil.save(os.path.join(save_dir, f'label{i}.jpg'))


