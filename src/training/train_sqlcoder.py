from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "genai-sqlcoder"

def train_sqlcoder(dataset, epochs=3):

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    model.train()

    for epoch in range(epochs):
        for sample in dataset:

            inputs = tokenizer(sample["question"], return_tensors="pt")
            labels = tokenizer(sample["sql"], return_tensors="pt").input_ids

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
