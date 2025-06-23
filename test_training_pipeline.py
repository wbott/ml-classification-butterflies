#!/usr/bin/env python3
"""End-to-end test for training pipeline."""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, '.')

from src.data.dataset import ButterflyDataset
from src.models.vgg_transfer import VGGTransferModel
from src.utils.config import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_sample_dataset(data_dir: Path, num_samples: int = 20):
    """Create a small sample dataset for testing."""
    
    # Create directories
    train_dir = data_dir / 'raw' / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images
    classes = ['MONARCH', 'SWALLOWTAIL', 'BLUE_MORPHO']
    filenames = []
    labels = []
    
    for i in range(num_samples):
        # Create a random colored image
        np.random.seed(i)  # For reproducibility
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        filename = f'butterfly_{i:03d}.jpg'
        img.save(train_dir / filename)
        
        # Add to lists
        filenames.append(filename)
        labels.append(classes[i % len(classes)])
    
    # Create CSV file
    df = pd.DataFrame({
        'filename': filenames,
        'label': labels
    })
    csv_path = data_dir / 'raw' / 'Training_set.csv'
    df.to_csv(csv_path, index=False)
    
    return csv_path, len(classes)


def test_training_pipeline():
    """Test the complete training pipeline."""
    
    print("🧪 Testing Training Pipeline")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Override config for testing
        config.set('data.batch_size', 4)
        config.set('training.epochs', 2)  # Very few epochs for testing
        config.set('paths.data_dir', str(temp_path))
        
        # Create sample dataset
        print("📊 Creating sample dataset...")
        csv_path, num_classes = create_sample_dataset(temp_path, num_samples=12)
        print(f"✅ Created dataset with {num_classes} classes")
        
        # Initialize dataset handler
        dataset = ButterflyDataset(str(temp_path))
        
        # Create data generators
        print("🔄 Creating data generators...")
        train_gen, val_gen = dataset.create_data_generators(
            train_csv=str(csv_path),
            validation_split=0.3,  # Use more validation for small dataset
            augment=False  # Disable augmentation for faster testing
        )
        
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Number of classes: {len(train_gen.class_indices)}")
        
        # Initialize model
        print("🏗️  Building VGG16 transfer model...")
        model = VGGTransferModel()
        input_shape = (224, 224, 3)
        model.build_model(input_shape, num_classes)
        model.compile_model()
        
        print("✅ Model built and compiled")
        
        # Train for a few steps
        print("🚀 Training model (2 epochs)...")
        try:
            history = model.train(
                train_generator=train_gen,
                validation_generator=val_gen,
                epochs=2  # Very short training for testing
            )
            
            print("✅ Training completed successfully!")
            
            # Check training history
            assert 'loss' in history
            assert 'accuracy' in history
            assert 'val_loss' in history
            assert 'val_accuracy' in history
            
            print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
        
        # Test predictions
        print("🔮 Testing predictions...")
        try:
            predictions = model.predict(val_gen)
            assert predictions.shape[0] == val_gen.samples
            assert predictions.shape[1] == num_classes
            
            # Check predictions are valid probabilities
            assert np.all(predictions >= 0)
            assert np.all(predictions <= 1)
            assert np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-5)
            
            print("✅ Predictions work correctly")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
        
        # Test model saving
        print("💾 Testing model saving...")
        try:
            model_path = temp_path / 'test_model.keras'
            model.save_model(str(model_path))
            assert model_path.exists()
            print("✅ Model saved successfully")
            
        except Exception as e:
            print(f"❌ Model saving failed: {e}")
            return False
        
        # Test fine-tuning setup
        print("🔧 Testing fine-tuning setup...")
        try:
            model.unfreeze_top_layers(2)
            print("✅ Fine-tuning setup successful")
            
        except Exception as e:
            print(f"❌ Fine-tuning setup failed: {e}")
            return False
    
    print()
    print("🎉 All tests passed! Training pipeline works correctly.")
    return True


if __name__ == '__main__':
    success = test_training_pipeline()
    sys.exit(0 if success else 1)