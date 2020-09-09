import logging
import pickle


def load_text(ifn):
    with open(ifn, "r") as f:
        txt = f.read()
    return txt


def save_text(text, ofn):
    with open(ofn, "w") as f:
        f.write(text)


def pkl_load(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def pkl_save(data, fn):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


LOG_LVLs = {
    'i': logging.INFO,
    'd': logging.DEBUG,
    'e': logging.ERROR,
    'w': logging.WARN
}


def create_logger(logger_name="", log_level="d", set_file=None):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(LOG_LVLs[log_level])
    if set_file:
        fh = logging.FileHandler(set_file)
        fh.setFormatter(formatter)
        fh.setLevel(LOG_LVLs[log_level])
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(LOG_LVLs[log_level])
        logger.addHandler(ch)

    return logger


class SeqEHRLogger:
    def __init__(self, logger_file=None, logger_level='i'):
        self.lf = logger_file
        self.lvl = logger_level

    def get_logger(self):
        return create_logger("SeqEHR", log_level=self.lvl, set_file=self.lf)
