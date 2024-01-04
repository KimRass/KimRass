from datetime import datetime
import logging
from pathlib import Path


class Logger:
    def get_logger(self):
        logger = logging.getLogger("logger")
        logger.propagate = False

        formatter = logging.Formatter(
			fmt="| %(asctime)s | %(levelname)s | %(message)s | %(filename)s | %(funcName)s() |",
			datefmt="%Y-%m-%d %H:%M:%S"
		)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.setLevel(logging.DEBUG)

        if self.out_dir:
            if self.append:
                file_handler = logging.FileHandler(
                    self.out_dir / f"logs_{datetime.now().strftime('%Y-%m-%d% %H:%M:%S')}.txt"
                )
            else:
                file_handler = logging.FileHandler(
                    self.out_dir / "logs.txt"
                )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def __init__(self, out_dir=None, append=False):
        out_dir = Path(out_dir)

        self.out_dir = out_dir
        self.append = append
