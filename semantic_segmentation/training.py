import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Accumulator, EarlyStopping


# TESTING: implement loss_function
def loss_function(
    pred_probs: torch.Tensor,
    gt_labels: torch.Tensor,
):
    loss_matrix = F.cross_entropy(input=pred_probs, target=gt_labels, reduction='none')
    return loss_matrix.mean(dim=(1, 2))


# TESTING: implment train
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
            optimizer.zero_grad()
            pred_probs: torch.Tensor = model(batch_images)
            pred_labels = pred_probs.max(dim=1).indices
            n_corrects = (pred_labels == gt_labels).sum()
            n_predictions = pred_labels.numel()
            loss = loss_function(pred_probs, gt_labels).mean()
            loss.backward()
            optimizer.step()
            
            # Accumulate the metrics
            train_metrics.add(n_correct=n_corrects, n_predictions=n_predictions, loss=loss)
            train_accuracy = train_metrics['n_correct'] / train_metrics['n_predictions']
            train_loss = train_metrics['loss'] / (batch + 1)
            print(
                f'Epoch {epoch + 1}/{n_epochs} || '
                f'Batch {batch + 1}/{len(train_dataset) // train_batch_size + 1} || '
                f'train_accuracy: {train_accuracy:.4e}, '
                f'train_loss: {train_loss:.2e}'
            )
        
        # Save checkpoint
        if checkpoint_output:
            torch.save(model, f'{checkpoint_output}/epoch{epoch + 1}.pt')

        # Reset metric records for next epoch
        train_metrics.reset()

        # TESTING: modify this after implement `evaluate`
        val_accuracy, val_loss = evaluate(model=model, dataset=val_dataset, batch_size=val_batch_size)
        print(f'Epoch {epoch + 1}/{n_epochs} || val_accuracy: {val_accuracy:.4e}, val_loss: {val_loss:.2e}')
        print('='*20)

        early_stopping(val_loss)
        if early_stopping:
            print('Early Stopped')
            break
    
    return model


# TESTING: modify `evaluate`
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
        n_corrects = (pred_labels == gt_labels).sum()
        n_predictions = pred_labels.numel()
        loss = loss_function(pred_probs, gt_labels).mean()

        # Accumulate the metrics
        metrics.add(n_corrects=n_corrects, n_predictions=n_predictions, loss=loss)

    # Compute the aggregate metrics
    accuracy = metrics['n_corrects'] / metrics['n_predictions']
    loss = metrics['loss'] / (batch + 1)
    return accuracy, loss


# TESTING: modify `predict`
def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    pred_probs: torch.Tensor = model(X)
    pred_labels = pred_probs.max(dim=1).indices
    return pred_labels



