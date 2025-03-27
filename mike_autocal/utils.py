import psutil
import os

def get_memory_usage():

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024