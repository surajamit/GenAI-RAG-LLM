from fastapi import FastAPI
from src.llm.custom_llm import CustomLLM

app = FastAPI(title="Enterprise GraphRAG API")
model = CustomLLM()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: str):
    response = model.generate(q)
    return {"response": response}
