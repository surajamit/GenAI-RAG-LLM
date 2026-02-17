class BenchmarkSuite:
    """
    Unified loader for all manuscript datasets.
    """

    def __init__(self):
        self.datasets = {}

    def load_all(self):
        from datasets import load_dataset

        self.datasets["msmarco"] = load_dataset("ms_marco", "v1.1")
        self.datasets["conll2003"] = load_dataset("conll2003")
        self.datasets["spider"] = load_dataset("spider")
        self.datasets["fever"] = load_dataset("fever", "v1.0")

        return self.datasets
