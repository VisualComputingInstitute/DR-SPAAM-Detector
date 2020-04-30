import os
import logging
from tensorboardX import SummaryWriter


def create_logger(root_dir, file_name='log.txt'):
    log_file = os.path.join(root_dir, file_name)
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_tb_logger(root_dir, tb_log_dir='tensorboard'):
    return SummaryWriter(log_dir=os.path.join(root_dir, tb_log_dir))
