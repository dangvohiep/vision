import os

import torch
import torchvision

from object_detection import show_boxes
from object_detection.datasets import BananasDataset, NuImagesDataset
from object_detection.training import train, predict
from object_detection.models import SingleShotDetection




# device: torch.device = torch.device('cpu')
# dataset = BananasDataset(is_train=True, device=device)
# net = SingleShotDetection(n_classes=dataset.n_classes).to(device=device)
# trainer = torch.optim.SGD(params=net.parameters(), lr=0.2, weight_decay=5e-4)
# net = train(model=net, dataset=dataset, optimizer=trainer, batch_size=32, n_epochs=20)
# torch.save(net, 'object_detection/trained_models/banana_detection.pt')

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




device: torch.device = torch.device('cpu')
dataset = NuImagesDataset(n_annotations=10, version='v1.0-mini', device=device)

net = SingleShotDetection(n_classes=dataset.n_categories).to(device=device)
trainer = torch.optim.SGD(params=net.parameters(), lr=0.2, weight_decay=5e-4)
net = train(model=net, dataset=dataset, optimizer=trainer, batch_size=32, n_epochs=20)
torch.save(net, 'object_detection/trained_models/nuimages_detection.pt')


X = torchvision.io.read_image(
    path=f'{os.environ["PYTHONPATH"]}/data/nuimages/sweeps/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537294835112404.jpg'
).to(device=device).float().unsqueeze(0)

output = predict(model=net, X=X).squeeze(0).detach().cpu()

anchors = []
for anchor_tensor in output:
    class_label = anchor_tensor[0].item()  # Extract class label
    score = anchor_tensor[1].item()  # Extract detection score
    if class_label == -1 or score <= 0.5:
        # Filter out anchors with class label -1 or score <= 0.5 (threshold level)
        continue
    anchors.append(anchor_tensor)
anchors = torch.stack(tensors=anchors, dim=0).cpu()



mapper =  {category['token']: float(i) for i, category in enumerate(dataset.nuim.category)}
show_boxes(
    input_image=X.squeeze(0).to(dtype=torch.uint8).cpu(),
    bboxes=anchors[:, 2:],  # bounding box coordinates.
    labels=[] * anchors.shape[0],  # Map class labels to strings
    output_path='object_detection/img/nuimages.jpg',
)


