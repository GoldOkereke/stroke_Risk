import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure logging to write to a file and the console
def setup_logging():
    logger = logging.getLogger() # Get the root logger
    logger.setLevel(logging.INFO) # Set the logging level to INFO (you can change this to DEBUG for more detailed logs)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ) # Define the log message format with timestamp, logger name, log level, and message
    
    console_handler = logging.StreamHandler(sys.stdout) # Create a console handler to output logs to the console
    console_handler.setFormatter(formatter) # Set the formatter for the console handler
    logger.addHandler(console_handler) # Add the console handler to the logger

    # Create a file handler to write logs to a file with the current date in the filename
    file_handler = logging.FileHandler(f"logs/app_{Path().stem}.log")
    file_handler.setFormatter(formatter) # Set the formatter for the file handler
    logger.addHandler(file_handler) # Add the file handler to the logger

    return logger # Return the configured logger instance

logger = setup_logging() # Initialize the logger when the module is imported

