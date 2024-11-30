import time
import logging
from .core.train import train_model, inference, loader_test

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store trained model
trained_model = None

def handle(event, context):
    global trained_model
    
    try:
        start_time = time.time()
        logger.info("Request started")
        
        if trained_model is None:
            logger.info("Starting model training...")
            train_start = time.time()
            trained_model = train_model()
            train_end = time.time()
            logger.info(f"Model training took {train_end - train_start:.2f} seconds")
        
        if event.method == "GET" and event.path == "/accuracy":
            inference_start = time.time()
            accuracy = inference(trained_model, loader_test)
            inference_end = time.time()
            logger.info(f"Inference took {inference_end - inference_start:.2f} seconds")
            
            total_time = time.time() - start_time
            logger.info(f"Total request processing time: {total_time:.2f} seconds")
            
            return {
                "statusCode": 200,
                "body": {
                    "accuracy": float(accuracy)
                }
            }
        
        return {
            "statusCode": 404,
            "body": {"error": "Not found"}
        }
            
    except Exception as e:
        logger.error(f"Error after {time.time() - start_time:.2f} seconds: {str(e)}", 
                    exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }