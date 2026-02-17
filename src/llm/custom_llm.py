class CustomLLM:
    def __init__(self, model_name="local-175b"):
        self.model_name = model_name

    def generate(self, prompt):
        # placeholder for local inference
        return "LLM response for: " + prompt[:120]
