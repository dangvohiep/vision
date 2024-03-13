import os

import torch
import torchvision

from object_detection import show_boxes
from object_detection.datasets import BananasDataset, NuImagesDataset
from object_detection.training import train, predict
from object_detection.model.ssd import SingleShotDetection




# device: torch.device = torch.device('cuda')
# dataset = BananasDataset(is_train=True, device=device)
# net = SingleShotDetection(n_classes=dataset.n_classes).to(device=device)
# optimizer = torch.optim.SGD(params=net.parameters(), lr=0.2, weight_decay=5e-4)
# net = train(
#     model=net, 
#     dataset=dataset, 
#     optimizer=optimizer, 
#     batch_size=32, 
#     n_epochs=20,
#     checkpoint_output=f'{os.environ["PYTHONPATH"]}/.checkpoint/banana_detection'   
# )
# torch.save(net, f'{os.environ["PYTHONPATH"]}/object_detection/trained_models/banana_detection.pt')

# X = torchvision.io.read_image(
#     path=f'{os.environ["PYTHONPATH"]}/object_detection/img/banana.jpg'
# ).to(device=device).float().unsqueeze(0)

# output = predict(model=net, X=X).squeeze(0).detach().cpu()

# anchors = []
# for anchor_tensor in output:
#     class_label = anchor_tensor[0].item()  # Extract class label
#     score = anchor_tensor[1].item()  # Extract detection score
#     if class_label == -1 or score <= 0.5:
#         # Filter out anchors with class label -1 or score <= 0.5 (threshold level)
#         continue
#     anchors.append(anchor_tensor)
# anchors = torch.stack(tensors=anchors, dim=0).cpu()

# mapper = {0: 'banana'}
# show_boxes(
#     input_image=X.squeeze(0).to(dtype=torch.uint8).cpu(),
#     bboxes=anchors[:, 2:],  # bounding box coordinates.
#     labels=[mapper[int(anchor[0])] for anchor in anchors],  # Map class labels to strings
#     output_path='img/banana_detection.jpg',
# )


device: torch.device = torch.device('cuda')
train_dataset = NuImagesDataset(n_annotations=10, version='v1.0-train', device=device)
val_dataset = NuImagesDataset(n_annotations=10, version='v1.0-val', device=device)

# net = SingleShotDetection(n_classes=train_dataset.n_categories).to(device=device)
net = torch.load(f'{os.environ["PYTHONPATH"]}/object_detection/trained_models/nuimages_detection_5.pt').to(device=device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
net = train(
    model=net, 
    train_dataset=train_dataset, 
    val_dataset=val_dataset,
    optimizer=optimizer, 
    train_batch_size=16, 
    val_batch_size=1, 
    n_epochs=20, 
    patience=5,
    tolerance=0.,
    checkpoint_output=f'{os.environ["PYTHONPATH"]}/.checkpoint/nuimages_detection'
)
torch.save(net, f'{os.environ["PYTHONPATH"]}/object_detection/trained_models/nuimages_detection_6.pt')

# device: torch.device = torch.device('cuda')
dataset = NuImagesDataset(n_annotations=10, version='v1.0-train', device=device)
net = torch.load(f'{os.environ["PYTHONPATH"]}/.checkpoint/nuimages_detection/epoch10.pt').to(device=device)

X = torchvision.io.read_image(
    path=f'{os.environ["PYTHONPATH"]}/data/nuimages/samples/CAM_FRONT/n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514865168561498.jpg'
).to(device=device).float().unsqueeze(0)

output = predict(model=net, X=X).squeeze(0).detach()

anchors = []
for anchor_tensor in output:
    class_label = anchor_tensor[0].item()  # Extract class label
    score = anchor_tensor[1].item()  # Extract detection score
    if class_label == -1 or score <= 0.15:
        # Filter out anchors with class label -1 or score <= 0.5 (threshold level)
        continue
    anchors.append(anchor_tensor)

anchors = torch.stack(tensors=anchors, dim=0).cpu()
show_boxes(
    input_image=X.squeeze(0).to(dtype=torch.uint8).cpu(),
    bboxes=anchors[:, 2:],  # bounding box coordinates.
    labels=[dataset.category_numeric_name_mapper[c.item()].split('.')[-1] for c in anchors[:, 0]],  # Map class labels to strings
    output_path='object_detection/img/nuimages.jpg',
)


