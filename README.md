# 🩺 FL4ME - Federated Learning for Medical Enclaves

Welcome to **FL4ME**! This project benchmarks **centralized** vs **federated learning** for breast cancer detection using the BreastMNIST dataset, with a focus on privacy and real-world clinical relevance.

---

## 🚀 Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/Fed-CAD.git
    cd Fed-CAD
    ```

2. **Install dependencies**
    - All dependencies are listed in `pyproject.toml`:
    ```bash
    pip install -e .
    ```

3. **Run experiments & analysis**
    - **Train & evaluate:**  
      Use `run_experiments.ipynb` to run experiments.
    - **Analyze results:**  
      Use `plots.ipynb` to generate visualizations, summary statistics, and key insights.
    - Results (CSVs, plots) will be saved in the `results/` directory.

---

## ⚙️ Modes of Operation

- **WandB Mode:**  
  Fetches live experiment data from [Weights & Biases](https://wandb.ai/) (requires authentication & project access).
- **Offline Mode:**  
  Uses exported CSV data (e.g., `results.csv`) for fully reproducible analysis.

---

## 📂 Repository Structure

```
Fed-CAD/
├── FL4ME/                # Source code for the Federated experiments
├── models/                # Saved model checkpoints
├── results/               # Exported results and analysis
├── run_experiments.ipynb  # Notebook to run experiments
├── plots.ipynb            # Main analysis and visualization notebook
├── pyproject.toml         # Python dependencies
├── README.md              # Project documentation
├── random_search_results.csv  # Hyperparameter search results
├── run_sweep.py           # Hyperparameter search script
└── sweep_config.yaml      # Configurations for the Hyperparameter search
```
