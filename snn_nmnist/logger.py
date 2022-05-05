import logging
import os
from types import MethodType


def init_logger(cfg):
    if not os.path.exists(f"./exp/{cfg.exp_id}"):
        os.makedirs(f"./exp/{cfg.exp_id}")

    filehandler = logging.FileHandler("./exp/{}/training.log".format(cfg.exp_id))
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info(
            "{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}".format(
                set=set, idx=idx, loss=loss, acc=acc
            )
        )

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


class DataLogger(object):
    """Average data logger."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt
