import os
import os
import time


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    """
    Create directory if it does not exist.
        path:           Path of directory.
    """
    if not os.path.exists(path):
        os.mkdir(path)

def log(log_file_path, string):
    """
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    """
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

