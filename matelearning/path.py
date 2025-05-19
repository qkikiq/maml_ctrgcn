import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
NTU_DATA_DIR = os.path.join(DATA_DIR, 'ntu')

# Define experiment directories
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

# Define paths for specific files
NTU60_CS_PATH = os.path.join(NTU_DATA_DIR, 'NTU60_CS.npz')
NTU60_LABEL_PATH = os.path.join(NTU_DATA_DIR, 'labels.txt')

# Define model checkpoint directory
CHECKPOINT_DIR = os.path.join(EXPERIMENTS_DIR, 'checkpoints')

def get_save_path(model_name):
    """
    Get path for saving model checkpoints
    """
    return os.path.join(CHECKPOINT_DIR, model_name)

def ensure_dirs():
    """
    Ensure all necessary directories exist
    """
    dirs = [DATA_DIR, NTU_DATA_DIR, EXPERIMENTS_DIR, CHECKPOINT_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)