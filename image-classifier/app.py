from flask import Flask
import logging
import time
from datetime import datetime
from function.core.train import train_model, inference, loader_test

# Configure logging format to include milliseconds
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Record total startup time
start_time = time.time()

# Record training start time
logger.info("Starting model training...")
train_start = time.time()

trained_model = train_model()

# Record training end time and calculate duration
train_end = time.time()
training_time = train_end - train_start
logger.info(f"Model training completed in {training_time:.2f} seconds")

# Record server startup time
server_start = time.time()
logger.info("Starting Waitress server on port 8080...")

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    accuracy = inference(trained_model, loader_test)
    return {'accuracy': float(accuracy)}

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
    
    # Calculate and log timing summary
    total_time = time.time() - start_time
    logger.info(f"""
    Timing Summary:
    - Training time: {training_time:.2f} seconds
    - Server startup time: {time.time() - server_start:.2f} seconds
    - Total startup time: {total_time:.2f} seconds
    """)