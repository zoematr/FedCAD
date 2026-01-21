import os
import argparse
import time
import numpy as np
import torch
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
    loss_fn = torch.nn.CrossEntropyLoss()

    # load all data centrally (partition_id=0, num_partitions=1)
    trainloader, testloader = load_data(0, 1)

    # W&B: log minimally and control model watching
    config = vars(args)
    config.update({
        "training_type": "centralized",
        "batch_size": trainloader.batch_size,
        "train_samples": len(trainloader.dataset),
        "test_samples": len(testloader.dataset),
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
    })
    
    run_name = f"central_lr{args.lr}_ep{args.epochs}"
    wandb.init(
        project="FedCAD",
        name=run_name,
        config=config,
        mode=args.wandb_mode,
        settings=wandb.Settings(_disable_stats=True),  # optional: disable system metrics
        tags=["centralized"],
        group="comparison"
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("test_acc", step_metric="epoch")
    if args.watch != "none":
        wandb.watch(model, log=args.watch, log_freq=args.watch_freq)

    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = running_loss / total_examples

        # log test metrics less frequently
        if epoch % args.eval_interval == 0:
            test_loss, test_acc = test(model, testloader, device)
            elapsed_time = (time.time() - start_time) / 60
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "elapsed_time_min": elapsed_time
            }, commit=True)
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.3f} | test_loss={test_loss:.3f} | test_acc={test_acc:.3f} | time={elapsed_time:.1f}min")
        else:
            wandb.log({"epoch": epoch, "train_loss": train_loss}, commit=True)
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.3f}")

    # Calculate detailed final metrics
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    total_time = time.time() - start_time
    test_loss, test_acc = test(model, testloader, device)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # ROC-AUC (binary classification)
    try:
        auc_roc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc_roc = 0.0
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_model_centralized.pt")
    print(f"\nSaved model -> models/final_model_centralized.pt")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Final metrics: acc={test_acc:.3f}, prec={precision:.3f}, rec={recall:.3f}, f1={f1:.3f}, auc={auc_roc:.3f}")

    # Summarize final metrics in wandb
    wandb.run.summary.update({
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "final_precision": precision,
        "final_recall": recall,
        "final_f1": f1,
        "final_auc_roc": auc_roc,
        "total_training_time_min": total_time / 60,
        "total_epochs": args.epochs,
    })

    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu", action="store_true", help="use GPU if available")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")

    # new controls
    p.add_argument("--eval-interval", type=int, default=1, help="log test metrics every N epochs")
    p.add_argument("--watch", choices=["none", "gradients", "parameters", "all"], default="none", help="W&B model watching")
    p.add_argument("--watch-freq", type=int, default=100, help="W&B watch log frequency (steps)")

    args = p.parse_args()
    train_central(args)