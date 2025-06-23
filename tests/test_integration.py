"""Integration tests for the reorganized project structure."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.models.vgg_transfer import VGGTransferModel
from src.data.dataset import ButterflyDataset


class TestProjectStructure:
    """Test the new project structure components."""
    
    def test_config_system(self):
        """Test configuration system works."""
        # Test getting values
        batch_size = config.get('data.batch_size')
        assert batch_size == 32
        
        # Test nested values
        vgg_units = config.get('models.vgg16.dense_units')
        assert vgg_units == 512
        
        # Test default values
        missing_value = config.get('nonexistent.key', 'default')
        assert missing_value == 'default'
    
    def test_vgg_model_creation(self):
        """Test VGG model can be created and compiled."""
        model = VGGTransferModel()
        
        # Build model
        keras_model = model.build_model(
            input_shape=(224, 224, 3),
            num_classes=10  # Smaller for testing
        )
        
        # Check model structure
        assert keras_model.input_shape == (None, 224, 224, 3)
        assert keras_model.output_shape == (None, 10)
        
        # Compile model
        model.compile_model()
        
        # Check model is compiled
        assert model.model.optimizer is not None
        assert model.model.loss is not None
    
    def test_model_prediction(self):
        """Test model can make predictions."""
        model = VGGTransferModel()
        model.build_model(input_shape=(224, 224, 3), num_classes=5)
        model.compile_model()
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8)
        dummy_input = dummy_input.astype(np.float32) / 255.0
        
        # Make prediction
        predictions = model.predict(dummy_input)
        
        # Check prediction shape
        assert predictions.shape == (2, 5)
        
        # Check predictions are valid probabilities
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-5)
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'filename': [f'butterfly_{i:03d}.jpg' for i in range(20)],
            'label': ['MONARCH', 'SWALLOWTAIL', 'BLUE_MORPHO'] * 6 + ['MONARCH', 'SWALLOWTAIL']
        }
        return pd.DataFrame(data)
    
    def test_dataset_creation(self, sample_csv_data):
        """Test dataset creation with CSV data."""
        dataset = ButterflyDataset()
        
        # Test metadata loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            df = dataset.load_metadata(csv_path)
            assert len(df) == 20
            assert 'filename' in df.columns
            assert 'label' in df.columns
            assert df['label'].nunique() == 3
        finally:
            os.unlink(csv_path)
    
    def test_model_callbacks(self):
        """Test model callbacks are created correctly."""
        model = VGGTransferModel()
        
        # Create temporary directory for model saving
        with tempfile.TemporaryDirectory() as temp_dir:
            callbacks = model.get_callbacks(temp_dir)
            
            # Check we have the expected callback types
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert 'EarlyStopping' in callback_types
            assert 'ReduceLROnPlateau' in callback_types
            assert 'ModelCheckpoint' in callback_types
    
    def test_config_modification(self):
        """Test configuration can be modified."""
        original_batch_size = config.get('data.batch_size')
        
        # Modify config
        config.set('data.batch_size', 64)
        assert config.get('data.batch_size') == 64
        
        # Restore original
        config.set('data.batch_size', original_batch_size)
        assert config.get('data.batch_size') == original_batch_size


class TestTrainingPipeline:
    """Test training pipeline components."""
    
    def test_model_training_setup(self):
        """Test model training can be set up (without actual training)."""
        model = VGGTransferModel()
        model.build_model(input_shape=(224, 224, 3), num_classes=3)
        model.compile_model()
        
        # Test model summary works
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed: {e}")
    
    def test_fine_tuning_setup(self):
        """Test fine-tuning setup works."""
        model = VGGTransferModel()
        model.build_model(input_shape=(224, 224, 3), num_classes=3)
        model.compile_model()
        
        # Test unfreezing layers
        model.unfreeze_top_layers(2)
        
        # Check that some VGG16 layers are now trainable
        vgg_layers = [layer for layer in model.model.layers if layer.name.startswith('block')]
        trainable_vgg_layers = [layer for layer in vgg_layers if layer.trainable]
        assert len(trainable_vgg_layers) >= 2
    
    def test_feature_extractor(self):
        """Test feature extractor creation."""
        model = VGGTransferModel()
        model.build_model(input_shape=(224, 224, 3), num_classes=3)
        
        # Get feature extractor
        feature_extractor = model.get_feature_extractor()
        
        # Check feature extractor shape
        assert feature_extractor.input_shape == (None, 224, 224, 3)
        # VGG16 with global average pooling should output 512 features
        assert feature_extractor.output_shape == (None, 512)