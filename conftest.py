import pytest
import numpy as np
import tensorflow as tf
import warnings
import os

# Configure TensorFlow to be less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Suppress common warnings during testing
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='tensorflow')

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility"""
    np.random.seed(42)
    tf.random.set_seed(42)

@pytest.fixture(scope="session")
def tensorflow_config():
    """Configure TensorFlow for testing"""
    # Limit GPU memory growth if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    yield
    
    # Clean up
    tf.keras.backend.clear_session()

@pytest.fixture
def sample_rgb_image():
    """Create a standard RGB image for testing"""
    np.random.seed(42)
    return np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)

@pytest.fixture
def sample_butterfly_classes():
    """Standard butterfly class names for testing"""
    return [
        'MONARCH', 'SWALLOWTAIL', 'BLUE_MORPHO', 'CABBAGE_WHITE', 'PAINTED_LADY',
        'RED_ADMIRAL', 'MOURNING_CLOAK', 'COMMON_BUCKEYE', 'ORANGE_SULPHUR', 'QUESTION_MARK'
    ]