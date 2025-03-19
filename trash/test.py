import logging
from pathlib import Path

file = Path("experiments") / "log.log"
logging.basicConfig(filemode='a', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Some:
    def __init__(self):
        logging.basicConfig(filename=file, filemode='a', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
        logging.info('Some text')
        
    def some(self):
        logging.info('Text from inside')
        
some = Some()
some.some()