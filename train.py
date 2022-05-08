import argparse
from PIL import Image
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms
from sklearn.metrics import classification_report
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a model to distinguish between different check boxes.')

parser.add_argument('-f', dest='model_path', default='resnet_18_weights.pt')
parser.add_argument('-d', dest='data_path', default='misc/data')


if __name__ == '__main__':
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: " + device)

    # Datasets for training and validation
    transform_composition = transforms.Compose([transforms.Resize((256,256)), transforms.RandomRotation(80),
                                                transforms.RandomResizedCrop((224,224)), transforms.ToTensor(), transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )])
    transform_composition_val = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )])
    base_ds = ImageFolder(args.data_path, transform=transform_composition)
    base_val_ds = ImageFolder(args.data_path, transform=transform_composition_val)

    # Splitting data into train, validate and test.
    train_ds, _ = torch.utils.data.random_split(base_ds, [int(0.7*len(base_ds)), len(base_ds) - int(0.7*len(base_ds))], torch.Generator().manual_seed(42))
    _, val_ds = torch.utils.data.random_split(base_val_ds, [int(0.7*len(base_val_ds)), len(base_val_ds) - int(0.7*len(base_val_ds))], torch.Generator().manual_seed(42))
    val_ds, test_ds = torch.utils.data.random_split(val_ds, [100, 55], torch.Generator().manual_seed(42))

    train_dl = torch.utils.data.DataLoader(train_ds, 32, True, num_workers=2)
    val_dl = torch.utils.data.DataLoader(val_ds, 32, False, num_workers=2)
    test_dl = torch.utils.data.DataLoader(test_ds, 32, False, num_workers=2)

    # Initialize a pretrained ResNet 18 Model
    model = resnet18(True)
    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last layer to output 3 classes
    model.fc = torch.nn.Linear(512, 3)
    # Unfreeze the last block of resnet for training.
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Move model to device
    model = model.to(device)
    # Define Optimizer and Loss functions
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1) # SGD because of generalizability
    criterion = torch.nn.CrossEntropyLoss()
    # Model Training
    for epoch in range(100):
        model.train()
        total_loss = 0
        for i, (batch_images, batch_labels) in tqdm(enumerate(train_dl), total=len(train_dl)):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            loss = criterion(model(batch_images), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print()
        print(f'Epoch {epoch:3} Train Loss: {total_loss/len(train_dl)}')

    # Validation Loop
    print("Validation Started!")
    with torch.inference_mode():
        model.eval()
        y_true = []
        y_pred = []
        total_loss = 0
        for i, (batch_images, batch_labels) in tqdm(enumerate(val_dl), total=len(val_dl)):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            y_true.extend(batch_labels.tolist())
            pred = model(batch_images)
            y_pred.extend(torch.argmax(torch.softmax(pred, 1), 1).tolist())
            loss = criterion(pred, batch_labels)
            total_loss += loss.item()

        print()
        print(classification_report(y_true, y_pred, target_names=base_ds.classes))
        print(f"Val Loss: {total_loss/len(val_dl)}")

    # Test set: model evaluation
    print("TESTING MODEL!")
    with torch.inference_mode():
        model.eval()
        y_true = []
        y_pred = []
        for i, (batch_images, batch_labels) in tqdm(enumerate(test_dl)):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            y_true.extend(batch_labels.tolist())
            pred = model(batch_images)
            y_pred.extend(torch.argmax(torch.softmax(pred, 1), 1).tolist())
            loss = criterion(pred, batch_labels)
        print()
        print(classification_report(y_true, y_pred, target_names=base_ds.classes))

    torch.save(model.to('cpu').state_dict(), args.model_path)