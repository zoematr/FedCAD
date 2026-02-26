"""FL4ME: A Flower / PyTorch app."""
from datetime import datetime
import time
import torch
import wandb
import numpy as np
import os
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx, DifferentialPrivacyServerSideFixedClipping

from FL4ME.task import Net, load_data, test

# Create ServerApp
app = ServerApp()


#python train_central.py --epochs 10
def save_to_csv(data: dict, csv_path: str = "results.csv"):
    """Save results to CSV file with consistent column order."""
    columns = [
        "name", "training_type", "state", "strategy",
        "lr", "epochs", "num_rounds", "local_epochs",
        "final_test_acc", "final_test_loss", "final_precision",
        "final_recall", "final_f1", "final_auc_roc",
        "total_training_time_min", "created_at"
    ]
    
    # Ensure all columns exist in data
    for col in columns:
        if col not in data:
            data[col] = None
    
    new_df = pd.DataFrame([data])[columns]
    
    if os.path.exists(csv_path):
        # Read existing, ensure same columns, append
        existing_df = pd.read_csv(csv_path)
        for col in columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        combined_df = pd.concat([existing_df[columns], new_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    #FedProx parameter
    proximal_mu: float = float(context.run_config.get("proximal-mu", 0.001))
    
    # DP parameters
    noise_multiplier: float = float(context.run_config.get("noise-multiplier", 0.1))
    clipping_norm: float = float(context.run_config.get("clipping-norm", 1.0))
    num_sampled_clients: int = int(context.run_config.get("num-sampled-clients", 5))
    use_wandb: bool = context.run_config.get("use-wandb", True)

    if context.run_config.get("fedprox", False):
        strategy = FedProx(
            proximal_mu=proximal_mu,
            fraction_train=fraction_train,
        )
        strategy_name = "fedprox"
    elif context.run_config.get("fedprox-dp", False):
        base_strategy = FedProx(
            proximal_mu=proximal_mu,
            fraction_train=fraction_train,
        )
        strategy = DifferentialPrivacyServerSideFixedClipping(
        strategy=base_strategy,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        num_sampled_clients=num_sampled_clients,
        )
        strategy_name = "fedprox-dp"
    else:
        strategy = FedAvg(fraction_train=fraction_train)
        strategy_name = "fedavg"

    # Load test data for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = load_data(0, 1)  # Use full dataset for testing

    # Initialize wandb with comprehensive config
    run_name = f"federated_lr{lr:.3f}_r{num_rounds}_le{local_epochs}"
    if use_wandb:
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        wandb.init(
            project="FL4ME",
            name=run_name,
            config={
                "training_type": "federated",
                "fraction_train": float(f"{fraction_train:.3f}"),
                "local_epochs": local_epochs,
                "strategy": strategy_name,
                "lr": float(f"{lr:.4f}"),
                "total_local_epochs": num_rounds * local_epochs,
                "test_samples": len(testloader.dataset),
                "noise_multiplier": noise_multiplier,
                "clipping_norm": clipping_norm,
                "num_sampled_clients": num_sampled_clients,
                "num_rounds": num_rounds,
                "proximal_mu": proximal_mu,
            },
            tags=["federated", "random_search"] if is_sweep else ["federated"],
            group="random_search" if is_sweep else "comparison"
        )
        wandb.define_metric("round")
        wandb.define_metric("train_loss", step_metric="round")
        wandb.define_metric("test_loss", step_metric="round")
        wandb.define_metric("test_acc", step_metric="round")
    else:
        print("⚠️ WandB logging is disabled. Set 'use_wandb' to True in run config to enable.")
    
    # Track start time
    start_time = time.time()

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize aggregation strategy


    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Evaluate final model with detailed metrics
    global_model.load_state_dict(result.arrays.to_torch_state_dict())
    global_model.to(device)
    global_model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = global_model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate final metrics
    total_time = time.time() - start_time
    test_loss, test_acc = test(global_model, testloader, device)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc_roc = 0.0
    
    print(f"\nFinal Results | test_loss={test_loss:.3f} | test_acc={test_acc:.3f}")
    print(f"Metrics: prec={precision:.3f}, rec={recall:.3f}, f1={f1:.3f}, auc={auc_roc:.3f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    
    # Summary metrics
    wandb.run.summary.update({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "final_precision": precision,
        "final_recall": recall,
        "final_f1": f1,
        "final_auc_roc": auc_roc,
        "total_training_time_min": total_time / 60,
        "total_rounds": num_rounds,
        "total_local_epochs": num_rounds * local_epochs,
    })

    save_to_csv({
        "name": run_name,
        "training_type": "federated",
        "state": "finished",
        "strategy": strategy_name,
        "lr": lr,
        "epochs": None,  # Not applicable for federated
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "final_precision": precision,
        "final_recall": recall,
        "final_f1": f1,
        "final_auc_roc": auc_roc,
        "total_training_time_min": total_time / 60,
        "created_at": created_at,
    })


    # Save final model to disk
    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    
    wandb.finish()
