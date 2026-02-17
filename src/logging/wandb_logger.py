# Weights & Biases Logging

import wandb

class WandBLogger:
    """
    Experiment tracking and reporting.
    """

    def __init__(self, project="genai-enterprise"):
        wandb.init(project=project)

    def log_metrics(self, metrics: dict, step=None):
        wandb.log(metrics, step=step)

    def log_table(self, name, dataframe):
        wandb.log({name: wandb.Table(dataframe=dataframe)})
