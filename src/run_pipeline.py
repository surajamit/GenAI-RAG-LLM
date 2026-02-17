"""
Full GraphRAG + ATS pipeline.
"""

from utils.config_loader import load_config
from utils.profiler import timer


def main():
    cfg = load_config("configs/experiment.yaml")

    with timer("pipeline_total"):
        print("Pipeline initialized with config:", cfg)


if __name__ == "__main__":
    main()
