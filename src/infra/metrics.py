import time

class MetricsCollector:
    """
    Tracks latency, throughput, uptime.
    """

    def __init__(self):
        self.start_time = time.time()
        self.requests = 0

    def log_request(self):
        self.requests += 1

    def uptime(self):
        return time.time() - self.start_time

    def throughput(self):
        elapsed = self.uptime()
        return self.requests / elapsed if elapsed > 0 else 0
