import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.models import resnet18
from torchgeo.datasets import EuroSAT
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import sys

torch.set_float32_matmul_precision('high')

class EuroSATClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def visualize_predictions(model, dataloader, class_names, num_images=6):

    # Plot metrics after training

    log_dir = trainer.logger.log_dir
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    def extract_values(events):
        return [e.value for e in events]

    def extract_steps(events):
        return [e.step for e in events]

    train_loss = ea.Scalars("train_loss")
    val_loss = ea.Scalars("val_loss")
    train_acc = ea.Scalars("train_acc")
    val_acc = ea.Scalars("val_acc")

    plt.figure()
    plt.plot(extract_steps(train_loss), extract_values(train_loss), label="Train Loss")
    plt.plot(extract_steps(val_loss), extract_values(val_loss), label="Val Loss")
    plt.plot(extract_steps(train_acc), extract_values(train_acc), label="Train Accuracy")
    plt.plot(extract_steps(val_acc), extract_values(val_acc), label="Val Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 6))

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"]
            y = batch["label"]
            preds = model(x)
            predicted = preds.argmax(dim=1)

            for i in range(x.size(0)):
                if images_shown >= num_images:
                    break

                try:
                    img = x[i][[3, 2, 1]].permute(1, 2, 0).cpu().numpy()
                except IndexError:
                    print("Input image has fewer than 8 bands.")
                    continue

                img = (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1)) + 1e-5)
                img = img.clip(0, 1)
                
                plt.subplot(2, num_images // 2, images_shown + 1)
                plt.imshow(img)
                plt.title(f"Predicted: {class_names[predicted[i]]}\nGround Truth: {class_names[y[i]]}")
                plt.axis("off")
                images_shown += 1
    plt.tight_layout()
    plt.show()
    sys.exit()

if __name__ == "__main__":
    dataset = EuroSAT(root="./data", download=True)
    # 1. Normalization would typically happen here
    # 2. Split dataset
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    # 3.  Dataloader setup
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True)

    # 4. Instantiate model with number of classes
    model = EuroSATClassifier(num_classes=len(dataset.classes))

    # 5. Train the model
    trainer = pl.Trainer(max_epochs=50, accelerator="auto", devices="auto")
    trainer.fit(model, train_loader, val_loader)

    # 6/ Visualize predictions
    visualize_predictions(model, val_loader, dataset.classes, num_images=6)
