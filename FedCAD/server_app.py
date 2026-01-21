"""FLfraud: A Flower / PyTorch app."""

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from FedCAD.task import Net, load_data, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    # Initialize wandb
    wandb.init(
        project="FedCAD",
        name="federated",
        config={
            "num_rounds": num_rounds,
            "fraction_train": float(f"{fraction_train:.3f}"),
            "local_epochs": local_epochs,
            "lr": float(f"{lr:.4f}"),
            "training_type": "federated"
        },
        tags=["federated"]
    )
    wandb.define_metric("round")
    wandb.define_metric("train_loss", step_metric="round")
    wandb.define_metric("test_loss", step_metric="round")
    wandb.define_metric("test_acc", step_metric="round")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Load test data for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = load_data(0, 1)  # Use full dataset for testing

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Evaluate and log metrics after each round
    for round_num in range(1, num_rounds + 1):
        # Get metrics from this round (if available in result)
        if hasattr(result, 'metrics_distributed') and result.metrics_distributed:
            round_metrics = result.metrics_distributed.get(round_num, {})
            
            # Aggregate training loss from clients
            if 'train_loss' in round_metrics:
                train_loss = round_metrics['train_loss']
                wandb.log({"round": round_num, "train_loss": train_loss}, commit=False)
                print(f"Round {round_num}/{num_rounds} | train_loss={train_loss:.3f}")

    # Evaluate final model
    global_model.load_state_dict(result.arrays.to_torch_state_dict())
    test_loss, test_acc = test(global_model, testloader, device)
    print(f"\nFinal Results | test_loss={test_loss:.3f} | test_acc={test_acc:.3f}")
    
    wandb.run.summary["final_test_loss"] = test_loss
    wandb.run.summary["final_test_acc"] = test_acc

    # Save final model to disk
    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    
    wandb.finish()
