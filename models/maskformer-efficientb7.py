import os
import torch
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from transformers import MaskFormerForInstanceSegmentation, MaskFormerConfig, MaskFormerImageProcessor
import albumentations as A
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load EfficientNet-B7 backbone
efficientnet_backbone = EfficientNet.from_pretrained("efficientnet-b7")

# MaskFormer Configuration
config = MaskFormerConfig(
    backbone_config=dict(
        model_name="efficientnet-b7",  # Specify EfficientNet backbone
        dilation=True
    )
)
model = MaskFormerForInstanceSegmentation(config)

# Load Image Processor
processor = MaskFormerImageProcessor()

# Set up the optimizer with learning rate 1e-4 and weight decay 1e-2
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# Data augmentation and transforms
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_val_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.3),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

# Example dataset and dataloader setup
class ImageSegmentationDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB"))
        instance_seg = np.array(self.dataset[idx]["annotation"])[..., 1]
        class_id_map = np.array(self.dataset[idx]["annotation"])[..., 0]
        class_labels = np.unique(class_id_map)
        inst2class = {i: label for i, label in enumerate(class_labels)}

        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed["image"], transformed["mask"]
            image = image.transpose(2, 0, 1)

        inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs

train_dataset = ImageSegmentationDataset(train_data, processor, transform=train_val_transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = ImageSegmentationDataset(val_data, processor, transform=train_val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set number of epochs
num_epochs = 2

for epoch in range(num_epochs):
    print(f"Epoch {epoch} | Training")
    model.train()
    train_loss = []

    for idx, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            print(f"Training loss: {round(sum(train_loss) / len(train_loss), 6)}")

    print(f"Average training loss: {sum(train_loss) / len(train_loss)}")

    # Validation loop with MAE and MSE calculation
    print(f"Epoch {epoch} | Validation")
    model.eval()
    val_loss = []
    all_labels = []
    all_predictions = []

    for idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
            loss = outputs.loss
            val_loss.append(loss.item())

            # Collect labels and predictions for MAE and MSE calculation
            for label in batch["mask_labels"]:
                all_labels.append(label.cpu().numpy())
            for prediction in outputs.logits:
                all_predictions.append(prediction.cpu().numpy())

    # Calculate MAE and MSE
    mae = mean_absolute_error(np.concatenate(all_labels), np.concatenate(all_predictions))
    mse = mean_squared_error(np.concatenate(all_labels), np.concatenate(all_predictions))

    print(f"Epoch {epoch} | Validation Loss: {sum(val_loss) / len(val_loss)} | MAE: {mae} | MSE: {mse}")

print("Training complete.")
