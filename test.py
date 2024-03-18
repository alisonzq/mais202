import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from efficientnet_pytorch import EfficientNet

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
dataset = ImageFolder(
    "garbage_classification", transform
)

class EfficientLite(pl.LightningModule):
    def __init__(self, lr: float, num_class: int, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_class)

        self.train_accuracy = MulticlassAccuracy(num_class)
        self.val_accuracy = MulticlassAccuracy(num_class)
        self.test_accuracy = MulticlassAccuracy(num_class)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        self.train_accuracy(torch.argmax(logits, dim=1), y)

        self.log("train_loss", loss.item(), on_epoch=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)

        self.val_accuracy(torch.argmax(logits, dim=1), y)

        self.log("val_loss", loss.item(), on_epoch=True)
        self.log(
            "val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)

        self.test_accuracy(torch.argmax(logits, dim=1), y)

        self.log("test_loss", loss.item(), on_epoch=True)
        self.log(
            "test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )

    def predict_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        return preds
    
model = EfficientLite(lr=0.00005, num_class=8)

test_border = len(dataset) - int(len(dataset) * (0.2))
indices = np.random.permutation(len(dataset)).tolist()
train_data = torch.utils.data.Subset(dataset, indices[:test_border])
validation_data = torch.utils.data.Subset(dataset, indices[test_border:])

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)

trainer = pl.Trainer(
    max_epochs=20,
    default_root_dir="models/",
)

trainer.fit(model, train_loader, validation_loader)

torch.save(model.state_dict(), 'bestmodel.pth')