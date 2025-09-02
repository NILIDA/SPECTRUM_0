import logging

def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()    
    
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - : %(message)s', datefmt='%H:%M:%S')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = configure_logger()
