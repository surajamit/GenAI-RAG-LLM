# LOCUST LOAD TEST (PRIMARY)
from locust import HttpUser, task, between
import random


class GenAIUser(HttpUser):

    wait_time = between(1, 3)

    @task(3)
    def qa_query(self):
        self.client.post("/generate", json={
            "query": "Explain GraphRAG architecture"
        })

    @task(2)
    def sql_query(self):
        self.client.post("/text2sql", json={
            "query": "List top employees by salary"
        })

    @task(1)
    def ats_analysis(self):
        self.client.post("/ats/evaluate", json={
            "resume": "sample resume text"
        })
