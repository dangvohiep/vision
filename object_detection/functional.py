from typing import List, Tuple, Callable, Union, Optional
from PIL.Image import Image

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.axes

import torch
import torchvision.transforms.functional as F


def generate_anchors(data: torch.Tensor, sizes: List[float], ratios: List[float]) -> torch.Tensor:
    """
    Generate anchor boxes with different shapes centered on each pixel of the input feature map.

    Parameters:
        - data (torch.Tensor): The input feature map with shape (batch_size, channels, height, width).
        - sizes (List[float]): A list of sizes (relative to the input feature map dimensions) for the anchor boxes.
        - ratios (List[float]): A list of aspect ratios (width / height) for the anchor boxes.

    Returns:
        - torch.Tensor: A tensor containing the coordinates of the generated anchor boxes for each pixel.
        The shape of the tensor is (batch_size, n_anchors, 4), where n_anchors is the total number of anchor
        boxes generated, and each anchor box is represented by 4 coordinates (xmin, ymin, xmax, ymax).

    Note:
        - `len(sizes) + len(ratios) - 1` anchor boxes are generated for each pixel of the input feature map. 
        Each box is centered at the pixel and has a shape determined by the specified sizes and ratios.
        - The sizes should be specified relative to the input feature map's dimensions, and the ratios
        determine the width and height of the boxes based on these sizes.
    """
    # Extract batch size:
    batch_size: int = data.shape[0]
    # Extract the height and width of the input feature map
    input_height: int = data.shape[-2]
    input_width: int = data.shape[-1]
    # Get the computation device, number of sizes and ratios
    device: torch.device = data.device
    num_sizes: int = len(sizes)
    num_ratios: int = len(ratios)
    # Calculate the number of boxes per pixel
    boxes_per_pixel: int = (num_sizes + num_ratios - 1)
    
    # Convert the sizes and ratios to tensors
    size_tensor: torch.Tensor = torch.tensor(sizes, device=device)
    ratio_tensor: torch.Tensor = torch.tensor(ratios, device=device)
    
    # Offsets to move the anchor to the center of a pixel
    offset_h: float = 0.5
    offset_w: float = 0.5
    # Scaled steps in y and x axes
    steps_h: float = 1.0 / input_height
    steps_w: float = 1.0 / input_width

    # Generate all center points for the anchor boxes
    center_h: torch.Tensor = (torch.arange(input_height, device=device) + offset_h) * steps_h
    center_w: torch.Tensor = (torch.arange(input_width, device=device) + offset_w) * steps_w
    shift_x: torch.Tensor
    shift_y: torch.Tensor
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_x: torch.Tensor = shift_x.reshape(-1)
    shift_y: torch.Tensor = shift_y.reshape(-1)

    # Generate widths and heights for anchor boxes
    sr_combination_1: torch.Tensor = size_tensor[0] * torch.sqrt(ratio_tensor[1:])
    sr_combination_2: torch.Tensor = size_tensor * torch.sqrt(ratio_tensor[0])
    w: torch.Tensor = torch.cat(tensors=(sr_combination_1, sr_combination_2), dim=0) * input_height / input_width

    sr_combination_1: torch.Tensor = size_tensor[0] / torch.sqrt(ratio_tensor[1:])
    sr_combination_2: torch.Tensor = size_tensor / torch.sqrt(ratio_tensor[0])
    h: torch.Tensor = torch.cat(tensors=(sr_combination_1, sr_combination_2), dim=0)
    
    # Compute half heights and half widths for anchor manipulations
    anchor_manipulations: torch.Tensor = torch.stack(tensors=(-w, -h, w, h)).t().repeat(input_height * input_width, 1) / 2

    # Generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid: torch.Tensor = torch.stack(tensors=[shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output: torch.Tensor = out_grid + anchor_manipulations

    # Repeat the tensor 100 times along the batch dimension, and return
    return output.unsqueeze(0).repeat(batch_size, 1, 1)


def box_corner_to_center(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from corner representation (upper-left, lower-right)
    to center representation (center x, center y, width, height).

    Parameters:
        - boxes (torch.Tensor): A tensor of shape (N, 4) containing N boxes, where each box is
        represented by its corner coordinates (x1, y1, x2, y2).

    Returns:
        - torch.Tensor: A tensor of shape (N, 4) where each box is represented by its center coordinates
        (center x, center y), width, and height.
    """
    # Unpack the corner coordinates
    x1: torch.Tensor = boxes[:, 0]
    y1: torch.Tensor = boxes[:, 1]
    x2: torch.Tensor = boxes[:, 2]
    y2: torch.Tensor = boxes[:, 3]
    # Calculate center x, center y, width, and height
    cx: torch.Tensor = (x1 + x2) / 2
    cy: torch.Tensor = (y1 + y2) / 2
    w: torch.Tensor = x2 - x1
    h: torch.Tensor = y2 - y1
    # Stack the new representation along the second dimension
    boxes: torch.Tensor = torch.stack((cx, cy, w, h), axis=1)
    return boxes


def box_center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from center representation (center x, center y, width, height)
    to corner representation (upper-left, lower-right).

    Parameters:
        - boxes (torch.Tensor): A tensor of shape (N, 4) containing N boxes, where each box is
        represented by its center coordinates (center x, center y), width, and height.

    Returns:
        - torch.Tensor: A tensor of shape (N, 4) where each box is represented by its corner coordinates
        (x1, y1, x2, y2).
    """
    # Unpack the center coordinates, width, and height
    cx: torch.Tensor = boxes[:, 0]
    cy: torch.Tensor = boxes[:, 1]
    w: torch.Tensor = boxes[:, 2]
    h: torch.Tensor = boxes[:, 3]
    # Calculate the corner coordinates
    x1: torch.Tensor = cx - 0.5 * w
    y1: torch.Tensor = cy - 0.5 * h
    x2: torch.Tensor = cx + 0.5 * w
    y2: torch.Tensor = cy + 0.5 * h
    # Stack the new representation along the second dimension
    boxes: torch.Tensor = torch.stack((x1, y1, x2, y2), axis=1)
    return boxes


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise Intersection over Union (IoU) between two sets of boxes.

    IoU is a measure of the overlap between two bounding boxes. This function
    calculates the IoU for each pair of boxes in `boxes1` and `boxes2`.

    Parameters:
        - boxes1 (torch.Tensor): A tensor of shape (N, 4) containing N boxes, where each box is
        represented by its (x1, y1, x2, y2) coordinates.
        - boxes2 (torch.Tensor): A tensor of shape (M, 4) containing M boxes, where each box is
        represented by its (x1, y1, x2, y2) coordinates.

    Returns: 
        - torch.Tensor: A tensor of shape (N, M) where each element [i, j] is the IoU of
        the i-th box in `boxes1` and the j-th box in `boxes2`.

    Note:
        - The coordinates of the boxes are expected to be (x1, y1, x2, y2), where
        (x1, y1) is the upper left corner, and (x2, y2) is the lower right corner.
    """
    # Function to calculate the area of boxes
    box_area: Callable[[torch.Tensor], torch.Tensor] = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Calculate the area of each box in both sets
    areas1: torch.Tensor = box_area(boxes1)   # shape = (N,)
    areas2: torch.Tensor = box_area(boxes2)   # shape = (M,)
    
    # Calculate intersections
    # Determine the coordinates of the intersection rectangles' upper left and lower right corners
    inter_upperlefts: torch.Tensor = torch.maximum(
        input=boxes1[:, None, :2],  # shape = (N, 1, 2)
        other=boxes2[:, :2]         # implicitly treated as shape = (1, M, 2)
    )   # shape = (N, M, 2)
    inter_lowerrights: torch.Tensor = torch.minimum(
        input=boxes1[:, None, 2:],  # shape = (N, 1, 2)
        other=boxes2[:, 2:]         # implicitly treated as shape = (1, M, 2)
    )   # shape = (N, M, 2)
    # Compute width and height of the intersections
    inters_wh: torch.Tensor = inter_lowerrights - inter_upperlefts # shape = (N, M, 2)
    # Ensure that intersection weigth and height are non-negative (happen when 2 boxes do not intersect)
    inters_wh: torch.Tensor = inters_wh.clamp(min=0)    # shape = (N, M, 2)
    # Calculate the area of the intersection rectangles
    inter_areas: torch.Tensor = inters_wh[:, :, 0] * inters_wh[:, :, 1] # shape = (N, M)
    # Calculate area of the unions
    union_areas: torch.Tensor = areas1[:, None] + areas2 - inter_areas    # shape = (N, M)
    # Compute IoU by dividing the intersection area by the union area
    return inter_areas / union_areas    # shape = (N, M)


def assign_bbox_to_anchor(
    gt_bboxes: torch.Tensor, 
    anchors: torch.Tensor, 
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Assign each anchor box to the closest ground-truth bounding box based on the Intersection over Union (IoU) metric.
    An anchor is assigned to a ground-truth box if their IoU exceeds a specified threshold, otherwise it is considered 
    background (assigned to -1)
    
    Parameters:
        - gt_bboxes (Tensor): A tensor of shape (N, 4) containing N ground-truth bounding boxes,
        where each box is represented by its (x1, y1, x2, y2) coordinates.
        - anchors (Tensor): A tensor of shape (M, 4) containing M anchor boxes, where each box is
        represented by its (x1, y1, x2, y2) coordinates.
        - iou_threshold (float, optional): The IoU threshold to use for matching anchors to ground-truth boxes.
        Defaults to 0.5.

    Returns:
        - Tensor: A tensor of shape (M,) where each element is the index of the ground-truth bounding box assigned
        to the corresponding anchor. If an anchor is not assigned to any ground-truth box, its value is -1.
    """
    n_gt_boxes: int = gt_bboxes.shape[0]     # N
    n_anchors: int = anchors.shape[0]        # M

    # Compute the IoU between all pairs of anchors and ground-truth boxes
    jaccard: torch.Tensor = compute_iou(boxes1=anchors, boxes2=gt_bboxes) # shape = (M, N)

    # Initialize the mapping of anchors to ground-truth boxes with -1 (unassigned)
    anchors_bbox_map: torch.Tensor = torch.full(
        size=(n_anchors,), fill_value=-1, dtype=torch.long, device=gt_bboxes.device
    )
    # Step 1: Assign each anchor to the ground-truth box with the highest IoU, if it exceeds the threshold
    max_ious: torch.Tensor
    indices: torch.Tensor
    max_ious, indices = torch.max(jaccard, dim=1)   # both have shape = (M,)
    # Index of all anchors having max_iou >= iou_threshold:
    anchor_i: torch.Tensor = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # shape = (K,) with K <= M
    # Index of according bounding boxes:
    bbox_j: torch.Tensor = indices[max_ious >= iou_threshold] # shape = (K,)
    # Map each anchor_i with one bbox_j
    anchors_bbox_map[anchor_i] = bbox_j     # shape = (M,)

    # Step 2: Ensure that each ground-truth box is assigned to at least one anchor
    # (the one with the highest IoU, regardless of iou_threshold)
    for _ in range(n_gt_boxes):
        # Find the anchor/ground-truth pair with the highest IoU
        max_idx: torch.Tensor = torch.argmax(jaccard)     # it flattens `jaccard` before find the index -> shape = ()
        bbox_idx: torch.Tensor = (max_idx % n_gt_boxes).long()      # column index
        anchor_idx: torch.Tensor = (max_idx // n_gt_boxes).long()   # row index
        anchors_bbox_map[anchor_idx] = bbox_idx     # map each anchor_i with one bbox_j
        # Prevent further assignment to the chosen ground-truth box and anchor
        jaccard[:, bbox_idx] = -1
        jaccard[anchor_idx, :] = -1

    return anchors_bbox_map


def compute_groundtruth(anchors: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process anchor boxes for a batch of N images, assigning class labels and calculating offsets 
    relative to the closest ground-truth bounding boxes.

    For each anchor box, this function determines whether it should be considered a positive example
    (i.e., assigned to a ground-truth box) or a negative example (i.e., considered background). It also
    calculates the offset between each positive anchor box and its corresponding ground-truth box.

    Parameters:
        - anchors (torch.Tensor): A tensor of shape (N, M, 4) containing the anchor boxes for each of N images,
        with M anchor boxes per image. Each anchor box is represented by its corner coordinates [x1, y1, x2, y2].
        - labels (torch.Tensor): A tensor of shape (N, G, 5) for N images, with G ground-truth objects per image.
        Each ground-truth box is represented by a class label and its corner coordinates [class, x1, y1, x2, y2].

    Returns:
        - Tuple containing three tensors:
        1. bbox_offset (torch.Tensor): Offsets (dx, dy, dw, dh) for each anchor box, shaped (N, M, 4).
        2. bbox_mask (torch.Tensor): A binary mask indicating which anchor boxes are positive examples,
           shaped (N, M, 4).
        3. class_labels (torch.Tensor): Class labels for each anchor box, where 0 indicates a background class,
           shaped (N, M).

    The function supports batch processing of multiple images, ensuring efficient computation across a dataset.
    
    """
    batch_size: torch.Tensor = anchors.shape[0]   # N
    n_anchors: torch.Tensor = anchors.shape[1]    # M

    batch_offsets: List[torch.Tensor] = []
    batch_masks: List[torch.Tensor] = []
    batch_class_labels: List[torch.Tensor] = []

    n: int
    for n in range(batch_size):
        image_labels: torch.Tensor = labels[n, :, :]      # shape = (G, 5)
        image_anchors: torch.Tensor = anchors[n, :, :]    # shape = (M, 4)
        # Assign each anchor box to a ground-truth bounding box
        anchors_bbox_map: torch.Tensor = assign_bbox_to_anchor(
            gt_bboxes=image_labels[:, 1:],  # shape = (G, 4)
            anchors=image_anchors,          # shape = (M, 4)
        )                                   # shape = (M,)
        # Create a mask to identify anchors that are assigned bounding box coordinates
        bbox_mask: torch.Tensor = (anchors_bbox_map >= 0).float()         # shape = (M,)
        # filter padding bounding boxes (set bbox_mask to 0 for padding bboxes)
        padding_indexes: torch.Tensor = (image_labels[:, 0] < 0).nonzero(as_tuple=True)[0]
        padding_mask: torch.Tensor = torch.isin(anchors_bbox_map, padding_indexes)
        bbox_mask[padding_mask] = 0
        bbox_mask: torch.Tensor = bbox_mask.unsqueeze(-1).repeat(1, 4)    # shape = (M, 4)
        # Initialize tensors for class labels and assigned bounding box coordinates
        class_labels: torch.Tensor = torch.zeros(n_anchors, dtype=torch.long, device=anchors.device)    # shape = (M,)
        assigned_bboxes: torch.Tensor = torch.zeros((n_anchors, 4), dtype=labels.dtype, device=labels.device)   # shape = (M, 4)

        # Update class labels and coordinates for assigned bounding boxes
        # Find the indices of anchors that are mapped with objects (not background)
        indices_true: torch.Tensor = torch.nonzero(anchors_bbox_map >= 0, as_tuple=False)   # shape = (K, 1) with G <= K <= M
        bbox_indices: torch.Tensor = anchors_bbox_map[indices_true]   # shape = (K, 1)
        # Increment class labels by 1 to account for background as class 0
        class_labels[indices_true] = image_labels[bbox_indices, 0].long() + 1 # shape = (K, 1), but class_labels still has shape = (M,)
        assigned_bboxes[indices_true] = image_labels[bbox_indices, 1:] # shape = (K, 1, 4), but assigned_bboxes still has shape = (M, 4)

        # Calculate offsets for anchor boxes relative to their assigned ground-truth boxes
        offset: torch.Tensor = offset_boxes(anchors=image_anchors, assigned_bboxes=assigned_bboxes) * bbox_mask # shape = (M, 4)

        # Append to the processed batch
        batch_class_labels.append(class_labels)
        batch_masks.append(bbox_mask)
        batch_offsets.append(offset)

    bbox_offset: torch.Tensor = torch.stack(batch_offsets, dim=0)
    bbox_mask: torch.Tensor = torch.stack(batch_masks, dim=0)
    class_labels: torch.Tensor = torch.stack(batch_class_labels, dim=0)

    return (bbox_offset, bbox_mask, class_labels)


def offset_boxes(anchors: torch.Tensor, assigned_bboxes: torch.Tensor) -> torch.Tensor:
    """
    Calculate the offsets needed to transform anchor boxes to match assigned ground-truth bounding boxes.
    The transformation is based on the center coordinates and the size of the boxes.

    Parameters:
        - anchors (Tensor): A tensor of shape (N, 4) containing N anchor boxes, where each box is
        represented by its corner coordinates (x1, y1, x2, y2).
        - assigned_bboxes (Tensor): A tensor of shape (N, 4) containing N ground-truth bounding boxes
        assigned to each anchor, represented by corner coordinates (x1, y1, x2, y2).

    Returns:
        - Tensor: A tensor of shape (N, 4) where each row contains the offsets (dx, dy, dw, dh)
        required to transform the corresponding anchor box to its assigned ground-truth box.
        Here, (dx, dy) are the center offsets, and (dw, dh) are the width and height offsets.
    """
    if anchors.shape[0] != assigned_bboxes.shape[0]:
        raise ValueError(
            "number of anchors must equal number of assigned bounding boxes, "
            f"got {anchors.shape[0]} and {assigned_bboxes.shape[0]}"
        )
    # Convert anchor and assigned boxes from corner to center format
    anchors: torch.Tensor = box_corner_to_center(anchors)
    assigned_bboxes: torch.Tensor = box_corner_to_center(assigned_bboxes)
    # Calculate offsets for center coordinates
    offset_xy: torch.Tensor = 10 * (assigned_bboxes[:, :2] - anchors[:, :2]) / anchors[:, 2:] # shape = (N, 2)
    # Calculate offsets for width and height
    # Use a small constant to ensure numerical stability
    offset_wh: torch.Tensor = 5 * torch.log(1e-6 + assigned_bboxes[:, 2:] / anchors[:, 2:])   # shape = (N, 2)
    # Concatenate the offset components to form the complete offset vector for each box
    offset: torch.Tensor = torch.cat([offset_xy, offset_wh], axis=1)  # shape = (N, 4)
    return offset


def offset_inverse(anchors: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Revert the offset predictions to predict bounding boxes based on anchor boxes.

    This function applies the inverse of the offset transformation used during model training,
    converting predicted offsets back into bounding box coordinates.

    Parameters:
        - anchors (torch.Tensor): A tensor of shape (N, 4) containing N anchor boxes, where each box is
        represented by its corner coordinates (x1, y1, x2, y2).
        - offsets (torch.Tensor): A tensor of shape (N, 4) containing N offsets for the
        anchor boxes. Offsets are in the format (dx, dy, dw, dh).

    Returns:
        - torch.Tensor: A tensor of shape (N, 4) containing the predicted bounding boxes, represented
        by their corner coordinates (x1, y1, x2, y2).

    The function converts anchor boxes from corner to center representation, applies the predicted
    offsets to scale and move the centers, and then converts the boxes back to corner representation.
    """
    # Convert anchor boxes from corner to center representation
    anchors: torch.Tensor = box_corner_to_center(anchors)
    # Apply offsets to the anchor boxes
    # Adjust position based on the predicted offsets (scaled by 10 for x, y)
    xy: torch.Tensor = (offsets[:, :2] * anchors[:, 2:] / 10) + anchors[:, :2]
    # Adjust size based on the predicted offsets (scaled by 5 for width, height)
    wh: torch.Tensor = torch.exp(offsets[:, 2:] / 5) * anchors[:, 2:]
    # Concatenate the adjusted position and size back into a single tensor
    boxes: torch.Tensor = torch.cat((xy, wh), axis=1)
    # Return the corner representation of the boxes
    return box_center_to_corner(boxes)


def non_maximum_supression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Perform Non-Maximum Suppression (NMS) on predicted bounding boxes.

    NMS filters out overlapping bounding boxes based on their Intersection over Union (IoU) scores,
    keeping only the bounding box with the highest confidence score in each group of overlapping boxes.

    Parameters:
    - boxes (torch.Tensor): A tensor of shape (N, 4) containing N predicted bounding boxes,
      where each box is represented by its corner coordinates (x1, y1, x2, y2).
    - scores (torch.Tensor): A tensor of shape (N,) containing confidence scores for each predicted bounding box.
    - iou_threshold (float): The IoU threshold for determining whether boxes overlap. Boxes with IoU
      above this threshold will be considered for suppression, keeping only the one with the highest score.

    Returns:
    - torch.Tensor: A tensor of shape (K,) containing the indices of K bounding boxes that are kept after NMS.

    The function sorts the bounding boxes by their confidence scores in descending order and iteratively
    removes boxes that overlap with a higher scoring box more than the specified IoU threshold.
    """
    # Sort boxes by their scores in descending order
    B: torch.Tensor = torch.argsort(scores, dim=-1, descending=True)  # contains the index for the boxes, shape = (N,)
    keep: List[torch.Tensor] = []

    # Iterate over boxes in order of descending score
    while B.numel() > 0:
        i: torch.Tensor = B[0]  # Index of the current box with the highest score, shape = ()
        keep.append(i)  # Always keep the current box
        if B.numel() == 1: break  # Stop if only one box is left
        # Compute IoU of the current box with all other boxes
        iou: torch.Tensor = compute_iou(
            boxes1=boxes[i, :].reshape(-1, 4),      # shape = (1, 4)
            boxes2=boxes[B[1:], :].reshape(-1, 4)   # shape = (N-1, 4)
        ).reshape(-1)   # shape (N-1,)
        # Find indices of boxes with IoU less than the threshold (these are not suppressed)
        inds: torch.Tensor = torch.nonzero(iou <= iou_threshold).reshape(-1)  # shape (K,) with K <= N-1
        # Update the list of boxes by removing suppressed boxes
        B: torch.Tensor = B[inds + 1]  # +1 adjusts indices since B[1:] was used for IoU calculation

    # Return indices of the boxes to be kept, ensuring the tensor is on the same device as the input boxes
    return torch.tensor(keep, device=boxes.device)


def filter_predictions(
    cls_probs: torch.Tensor, 
    pred_offsets: torch.Tensor, 
    anchors: torch.Tensor, 
    nms_threshold: float = 0.5, 
    pos_threshold: float = 0.009999999,
) -> torch.Tensor:
    """
    Perform bounding box predictions on a batch of images and filter them usinmg Non-Maximum Suppression (NMS).

    Reference: https://d2l.ai/chapter_computer-vision/anchor.html

    Parameters:
    - cls_probs (torch.Tensor): Class probabilities for each anchor box, of shape (batch_size, n_anchors, n_classes + 1).
      The first class is assumed to be the background.
    - offset_preds (torch.Tensor): Predicted offsets for each anchor box, of shape (batch_size, n_anchors, 4).
    - anchors (torch.Tensor): The anchor boxes, of shape (batch_size, n_anchors, 4).
    - nms_threshold (float): The IoU threshold for Non-Maximum Suppression.
    - pos_threshold (float): The threshold for considering a prediction as positive (non-background).

    Returns:
    - torch.Tensor: A tensor containing the final predictions for each image in the batch, including class IDs, confidence scores, 
      and bounding box coordinates, of shape (batch_size, n_anchors, 6). Class IDs of -1 indicate background or 
      suppressed boxes.
      
    """
    batch_size: int = cls_probs.shape[0]
    n_anchors: int = cls_probs.shape[1]

    batch_results: List[torch.Tensor] = []

    i: int
    for i in range(batch_size):
        # Process each image in the batch
        image_anchors: torch.Tensor = anchors[i]
        cls_probs: torch.Tensor = cls_probs[i]
        pred_offsets: torch.Tensor = pred_offsets[i]

        # Find the maximum class probability (excluding background) and its index (class ID)
        confidence_scores: torch.Tensor
        class_id: torch.Tensor
        confidence_scores, class_id = torch.max(cls_probs[:, 1:], dim=1)  # Exclude background class

        # Predict bounding boxes using the inverse offset transformation
        predicted_bboxes: torch.Tensor = offset_inverse(anchors=image_anchors, offsets=pred_offsets)

        # Apply Non-Maximum Suppression to filter overlapping boxes
        # Assuming non_maximum_suppression function supports batch processing and is adapted accordingly
        keep: torch.Tensor = non_maximum_supression(
            boxes=predicted_bboxes, scores=confidence_scores, iou_threshold=nms_threshold
        )
        # Initialize arrays to mark all as background initially
        final_class_ids: torch.Tensor = torch.full(
            (n_anchors,), -1, dtype=torch.long, device=anchors.device
        )
        final_confidence_scores: torch.Tensor = torch.zeros(
            (n_anchors,), dtype=torch.float32, device=anchors.device
        )
        final_bboxes: torch.Tensor = torch.zeros(
            (n_anchors, 4), dtype=torch.float32, device=anchors.device
        )
        # Update values based on NMS results
        final_class_ids[keep] = class_id[keep]
        final_confidence_scores[keep] = confidence_scores[keep]
        final_bboxes[keep] = predicted_bboxes[keep]

        # Filter out low-confidence predictions
        below_min_idx = final_confidence_scores < pos_threshold
        final_class_ids[below_min_idx] = -1  # Mark as background
        final_confidence_scores[below_min_idx] = 1 - final_confidence_scores[below_min_idx]  # Adjust confidence

        # Combine class IDs, confidence scores, and bounding boxes for the current image
        result: torch.Tensor = torch.cat(
            tensors=[
                final_class_ids.unsqueeze(-1).float(), 
                final_confidence_scores.unsqueeze(-1), 
                final_bboxes
            ], 
            dim=1,
        )
        batch_results.append(result.unsqueeze(0))  # Add batch dimension back

    # Concatenate results for all images in the batch
    return torch.cat(batch_results, dim=0)


def show_boxes(
    input_image: Union[Image, torch.Tensor],
    bboxes: torch.Tensor,
    labels: List[str],
    output_path: Optional[str],
) -> None:
    """
    Load an image from a file path, draw bounding boxes with optional labels and colors,
    and either display the image or save it to a file.

    Parameters:
        - input_image: the input PIL.Image object or a torch.Tensor of shape (C, H, W).
        - bboxes: an (N, 4) tensor representing relative positions of bounding boxes, where each is 
        [x_min, y_min, x_max, y_max], and N is the number of bouding boxes.
        - labels: List of labels for each bounding box
        - output_path: Path to save the output image.
    
    Returns:
        - None: The function saves the modified image to `output_path`.
    """

    if isinstance(input_image, torch.Tensor):
        input_image = F.to_pil_image(pic=input_image)

    if not output_path:
        output_path = f'{input_image}_bboxes.png'

    bboxes: torch.Tensor = bboxes.clone() # does not alter the input tensor in place
    bboxes[:, [0, 2]] *= input_image.size[0]
    bboxes[:, [1, 3]] *= input_image.size[1]

    labels += [''] * max(bboxes.shape[0] - len(labels), 0)
    labels: List[str] = labels[:bboxes.shape[0]]

    # Load the image
    _, ax = plt.subplots()
    ax.imshow(input_image)

    colors: List[str] = ['b', 'g', 'r', 'm', 'c']

    for i, bbox in enumerate(bboxes):
        color: str = colors[i % len(colors)]
        # Create a rectangle patch for each bounding box with the specified color
        rect = matplotlib.patches.Rectangle(
            xy=(bbox[0], bbox[1]), 
            width=bbox[2] - bbox[0], 
            height=bbox[3] - bbox[1],
            linewidth=1, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        # Add label text inside the bounding box if labels are provided
        text_color: str = 'k' if color == 'w' else 'w'
        ax.text(
            x=bbox[0], 
            y=bbox[1], 
            s=labels[i], 
            va='center', 
            ha='center',
            fontsize=9, 
            color=text_color, 
            bbox={'facecolor': color, 'lw': 0}
        )

    # Save the image
    plt.savefig(output_path)
    plt.close()


