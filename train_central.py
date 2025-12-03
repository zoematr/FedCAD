import os
import argparse
import torch
import torch.nn.functional as F
import wandb
from torchvision.transforms import Compose, ToTensor, Normalize
from medmnist import BreastMNIST
from torch.utils.data import DataLoader
from FedCAD.task import Net, load_data, test


#run with:
#python train_central.py --epochs 10


def train_central(args):

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # load all data centrally (partition_id=0, num_partitions=1)
    trainloader, testloader = load_data(0, 1)

    # set up wandb
    wandb.init(project="FedCAD", name="centralized", config=vars(args), mode=args.wandb_mode)

    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    ds = BreastMNIST(split="train", download=True, transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    images, labels = next(iter(loader))
    print(images.shape, labels.shape)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels_idx = labels.squeeze().long()
            labels_onehot = F.one_hot(labels_idx, num_classes=2).float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels_onehot)
            loss.backward()
            optimizer.step()

            batch_size = labels_onehot.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = running_loss / total_examples
        test_loss, test_acc = test(model, testloader, device)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_model_centralized.pt")
    print("Saved model -> models/final_model_centralized.pt")
    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu", action="store_true", help="use GPU if available")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = p.parse_args()
    train_central(args)