import redis
import json

class RedisCache:
    def __init__(self, host="localhost", port=6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def get(self, key):
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key, value, ttl=3600):
        self.client.setex(key, ttl, json.dumps(value))
