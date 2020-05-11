import os

import pandas as pd
import numpy as np

import albumentations
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.model_selection import train_test_split

from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader


class DenseCrossEntropy(nn.Module):
    # Taken from: https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()


class Model(nn.Module):
    # Modified from: https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.out = nn.Linear(in_features, 4)

    def forward(self, image, targets=None):
        batch_size, C, H, W = image.shape

        x = self.base_model.conv1(image)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = self.out(x)

        loss = None
        if targets is not None:
            loss = DenseCrossEntropy()(x, targets.type_as(x))

        return x, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str,
    )
    parser.add_argument(
        "--device", type=str,
    )
    parser.add_argument(
        "--epochs", type=int,
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    images = df.image_id.values.tolist()
    images = [os.path.join(args.data_path, "images", i + ".jpg") for i in images]
    targets = df[["healthy", "multiple_diseases", "rust", "scab"]].values

    model = Model()
    model.to(args.device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets
    )

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=(128, 128),
        augmentations=aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(128, 128),
        augmentations=aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)

    for epoch in range(args.epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=args.device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=args.device
        )
        predictions = np.vstack((predictions))
        roc_1 = metrics.roc_auc_score(valid_targets[:, 0], predictions[:, 0])
        roc_2 = metrics.roc_auc_score(valid_targets[:, 1], predictions[:, 1])
        roc_3 = metrics.roc_auc_score(valid_targets[:, 2], predictions[:, 2])
        roc_4 = metrics.roc_auc_score(valid_targets[:, 3], predictions[:, 3])
        mean_roc = (roc_1 + roc_2 + roc_3 + roc_4) / 4
        print(
            f"Epoch={epoch}, Train Loss={train_loss} Valid Loss={valid_loss}, Mean ROC AUC={mean_roc}"
        )

    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    images = test_df.image_id.values.tolist()
    images = [os.path.join(args.data_path, "images", i + ".jpg") for i in images]
    targets = np.zeros((len(images), 4))

    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    test_dataset = ClassificationLoader(
        image_paths=images, targets=targets, resize=(128, 128), augmentations=aug
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    predictions = Engine.predict(test_loader, model, device=args.device)
    predictions = np.vstack((predictions))

    sample = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
    sample.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = predictions
    sample.to_csv(os.path.join(args.data_path, "submission.csv"), index=False)
