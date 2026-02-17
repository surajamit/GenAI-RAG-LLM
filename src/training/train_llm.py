#LLM Fine-Tuning (with LoRA hooks)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "genai-llm-base"

def build_lora_model():

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # LoRA configuration (as per manuscript)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model


def train_llm(train_loader, epochs=3, lr=5e-5):

    model = build_lora_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
