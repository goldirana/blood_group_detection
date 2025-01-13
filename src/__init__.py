import logging
import os


log_dir = "logs"
log_file = "running.logs"

# create log dir
os.makedirs(log_dir, exist_ok=True)

# define handlers
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, log_file))

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 

# Create log config
logging.basicConfig(
    level=logging.DEBUG,
    format=formatter,
    handlers=[stream_handler, file_handler]
)

logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("\033[1;31m This is test\033[0m") # highlighting it with color code
    logger.error("This is error")