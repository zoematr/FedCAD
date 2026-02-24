import subprocess
import wandb

def main():
    # Get config from wandb sweep (but don't create a new run)
    wandb.init()
    config = wandb.config
    wandb.finish()  # Close immediately - server_app will create its own run
    
    # Build run-config string
    run_config = (
        f"lr={config.lr} "
        f"local-epochs={config.local_epochs} "
        f"num-server-rounds={config.num_server_rounds} "
        f"proximal-mu={config.proximal_mu} "
        f"noise-multiplier={config.noise_multiplier} "
        f"clipping-norm={config.clipping_norm}"
    )
    
    # Run flower - server_app.py will handle wandb
    cmd = ["flwr", "run", ".", "--run-config", run_config]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()