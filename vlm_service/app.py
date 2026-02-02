import json
import logging
import redis
import torch
import os
from PIL import Image
from config import Config
from utils import VLMAnalyzer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VLM_Service")

def main():
    analyzer = VLMAnalyzer()
    r = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)
    logger.info("VLM Service is ready and listening...")

    while True:
        _, data = r.brpop(Config.INPUT_QUEUE)
        event = json.loads(data.decode('utf-8'))
        
        result = analyzer.process_event(event)
        if result:
            logger.info(f"Analyzing Event: {event.get('event_id')}")
            logger.info(f"VLM Result: {result}")

            if "EOS" in event.get('event_id', ''):
                logger.info(f"End of Stream (EOS)")
                break

if __name__ == "__main__":
    main()