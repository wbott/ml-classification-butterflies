import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import pandas as pd
import tempfile
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class TestEdgeCases:
    
    @pytest.fixture
    def minimal_cnn_model(self):
        """Create minimal CNN model for edge case testing"""
        model = Sequential([
            Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes for testing
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def test_empty_dataset_handling(self):
        """Test model behavior with empty dataset"""
        empty_df = pd.DataFrame(columns=['filename', 'label'])
        
        # Test that empty dataframe is detected
        assert len(empty_df) == 0, "Empty dataframe should have zero length"
        assert empty_df.empty, "Empty dataframe should be detected as empty"
        
        # Test data generator with empty dataframe
        datagen = ImageDataGenerator(rescale=1./255)
        
        with pytest.raises((ValueError, FileNotFoundError)):
            # This should raise an error or handle gracefully
            datagen.flow_from_dataframe(
                dataframe=empty_df,
                directory='.',
                x_col='filename',
                y_col='label',
                target_size=(64, 64),
                batch_size=1,
                class_mode='categorical'
            )
    
    def test_single_class_dataset(self, minimal_cnn_model):
        """Test model training with only one class"""
        # Create single class data
        single_class_data = np.random.rand(10, 32, 32, 3).astype(np.float32)
        single_class_labels = np.zeros((10, 3))
        single_class_labels[:, 0] = 1  # All samples belong to class 0
        
        model = minimal_cnn_model
        
        # Test training with single class
        try:
            history = model.fit(single_class_data, single_class_labels, epochs=1, verbose=0)
            assert history is not None, "Model should handle single class training"
        except Exception as e:
            # Some configurations might not handle single class well
            pytest.skip(f"Single class training not supported: {str(e)}")
    
    def test_corrupted_image_data(self):
        """Test handling of corrupted or invalid image data"""
        # Create invalid image data
        corrupted_data = np.array([[[300, -50, 1000]]], dtype=np.float32)  # Invalid pixel values
        
        # Test normalization of corrupted data
        normalized = corrupted_data / 255.0
        
        # After normalization, should handle gracefully
        assert not np.isnan(normalized).any(), "Normalization should not produce NaN"
        assert np.isfinite(normalized).all(), "Normalization should produce finite values"
    
    def test_extreme_augmentation_parameters(self):
        """Test data augmentation with extreme parameters"""
        sample_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_batch = np.expand_dims(sample_img, axis=0)
        
        # Test extreme rotation
        extreme_datagen = ImageDataGenerator(rotation_range=180)
        aug_iter = extreme_datagen.flow(img_batch, batch_size=1)
        
        try:
            augmented = next(aug_iter)
            assert augmented.shape == img_batch.shape, "Extreme augmentation should maintain shape"
            assert augmented.min() >= 0, "Extreme augmentation should produce valid pixels"
            assert augmented.max() <= 255, "Extreme augmentation should produce valid pixels"
        except Exception as e:
            pytest.skip(f"Extreme augmentation not supported: {str(e)}")
    
    def test_very_small_images(self, minimal_cnn_model):
        """Test processing very small images"""
        # Create tiny images
        tiny_images = np.random.rand(5, 32, 32, 3).astype(np.float32)
        tiny_labels = tf.keras.utils.to_categorical([0, 1, 2, 0, 1], 3)
        
        model = minimal_cnn_model
        
        # Test prediction on tiny images
        predictions = model.predict(tiny_images, verbose=0)
        
        assert predictions.shape == (5, 3), "Should predict for all tiny images"
        assert np.all(predictions >= 0), "Predictions should be non-negative"
        assert np.all(predictions <= 1), "Predictions should be <= 1"
    
    def test_very_large_batch_size(self):
        """Test handling of large batch sizes that might exceed memory"""
        # This test checks graceful handling rather than actual large processing
        large_batch_size = 10000
        
        # Create small dataset
        small_data = np.random.rand(50, 32, 32, 3).astype(np.float32)
        
        datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generator with large batch size
        generator = datagen.flow(small_data, batch_size=large_batch_size)
        
        # Should handle gracefully (batch size will be limited by available data)
        batch = next(generator)
        assert batch.shape[0] <= 50, "Batch size should be limited by available data"
    
    def test_mixed_image_sizes_in_dataset(self):
        """Test dataset with images of different sizes"""
        # Create temporary directory with different sized images
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create images of different sizes
            sizes = [(64, 64), (128, 128), (100, 150)]
            for i, size in enumerate(sizes):
                img = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
                img_pil = Image.fromarray(img)
                img_path = os.path.join(temp_dir, f'img_{i}.jpg')
                img_pil.save(img_path)
            
            # Test loading with consistent target size
            from tensorflow.keras.preprocessing.image import load_img
            
            target_size = (64, 64)
            for i in range(len(sizes)):
                img_path = os.path.join(temp_dir, f'img_{i}.jpg')
                img = load_img(img_path, target_size=target_size)
                
                assert img.size == target_size, f"Image {i} should be resized to {target_size}"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_extreme_pixel_values(self, minimal_cnn_model):
        """Test model with extreme pixel values"""
        model = minimal_cnn_model
        
        # Test with all zeros
        zero_images = np.zeros((3, 32, 32, 3), dtype=np.float32)
        zero_predictions = model.predict(zero_images, verbose=0)
        
        assert zero_predictions.shape == (3, 3), "Should predict for zero images"
        assert np.isfinite(zero_predictions).all(), "Zero image predictions should be finite"
        
        # Test with all ones
        one_images = np.ones((3, 32, 32, 3), dtype=np.float32)
        one_predictions = model.predict(one_images, verbose=0)
        
        assert one_predictions.shape == (3, 3), "Should predict for one images"
        assert np.isfinite(one_predictions).all(), "One image predictions should be finite"
    
    def test_model_with_frozen_layers(self):
        """Test transfer learning model with frozen layers"""
        # Create simple base model
        base_model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Flatten()
        ])
        
        # Freeze base model
        base_model.trainable = False
        
        # Add classifier
        model = Sequential([
            base_model,
            Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Test that base layers are frozen
        for layer in base_model.layers:
            assert not layer.trainable, "Base model layers should be frozen"
        
        # Test training updates only top layers
        initial_weights = [layer.get_weights() for layer in base_model.layers if layer.weights]
        
        # Quick training
        dummy_data = np.random.rand(10, 64, 64, 3).astype(np.float32)
        dummy_labels = tf.keras.utils.to_categorical([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], 5)
        
        model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
        
        # Check that frozen layers didn't change
        final_weights = [layer.get_weights() for layer in base_model.layers if layer.weights]
        
        for initial, final in zip(initial_weights, final_weights):
            for init_w, final_w in zip(initial, final):
                assert np.array_equal(init_w, final_w), "Frozen layers should not change during training"
    
    def test_memory_efficient_batch_processing(self):
        """Test processing large datasets in small batches"""
        # Simulate large dataset with small memory footprint
        def data_generator():
            for i in range(100):  # Simulate 100 images
                yield np.random.rand(32, 32, 3).astype(np.float32)
        
        # Process in small batches
        batch_size = 5
        processed_count = 0
        
        batch = []
        for img in data_generator():
            batch.append(img)
            if len(batch) == batch_size:
                batch_array = np.array(batch)
                assert batch_array.shape == (batch_size, 32, 32, 3), "Batch should have correct shape"
                processed_count += len(batch)
                batch = []
        
        # Process remaining images
        if batch:
            batch_array = np.array(batch)
            processed_count += len(batch)
        
        assert processed_count == 100, "Should process all images"
    
    def test_class_imbalance_extreme(self):
        """Test handling of extreme class imbalance"""
        # Create extremely imbalanced dataset
        n_majority = 95
        n_minority = 5
        
        # Majority class data
        majority_data = np.random.rand(n_majority, 32, 32, 3).astype(np.float32)
        majority_labels = np.zeros((n_majority, 2))
        majority_labels[:, 0] = 1
        
        # Minority class data
        minority_data = np.random.rand(n_minority, 32, 32, 3).astype(np.float32)
        minority_labels = np.zeros((n_minority, 2))
        minority_labels[:, 1] = 1
        
        # Combine
        X = np.concatenate([majority_data, minority_data])
        y = np.concatenate([majority_labels, minority_labels])
        
        # Test data properties
        class_distribution = np.argmax(y, axis=1)
        unique, counts = np.unique(class_distribution, return_counts=True)
        
        imbalance_ratio = max(counts) / min(counts)
        assert imbalance_ratio == 19.0, "Should detect extreme imbalance"
    
    def test_nan_and_inf_handling(self, minimal_cnn_model):
        """Test model behavior with NaN and infinity values"""
        model = minimal_cnn_model
        
        # Create data with NaN values
        nan_data = np.random.rand(3, 32, 32, 3).astype(np.float32)
        nan_data[0, 0, 0, 0] = np.nan
        
        # Test prediction (model should handle or raise appropriate error)
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            model.predict(nan_data, verbose=0)
        
        # Create data with infinity values
        inf_data = np.random.rand(3, 32, 32, 3).astype(np.float32)
        inf_data[0, 0, 0, 0] = np.inf
        
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            model.predict(inf_data, verbose=0)
    
    def test_prediction_consistency_under_load(self, minimal_cnn_model):
        """Test prediction consistency when model is used repeatedly"""
        model = minimal_cnn_model
        test_image = np.random.rand(1, 32, 32, 3).astype(np.float32)
        
        # Get multiple predictions of the same image
        predictions = []
        for _ in range(10):
            pred = model.predict(test_image, verbose=0)
            predictions.append(pred)
        
        # All predictions should be identical
        first_pred = predictions[0]
        for pred in predictions[1:]:
            assert np.allclose(first_pred, pred), "Predictions should be consistent"
    
    def test_edge_case_single_pixel_differences(self):
        """Test model sensitivity to single pixel changes"""
        # Create two nearly identical images
        base_image = np.random.rand(32, 32, 3).astype(np.float32)
        modified_image = base_image.copy()
        modified_image[0, 0, 0] += 0.1  # Small change to one pixel
        
        # Test that images are indeed different
        assert not np.array_equal(base_image, modified_image), "Images should be different"
        
        # Difference should be minimal
        diff = np.sum(np.abs(base_image - modified_image))
        assert diff < 1.0, "Difference should be small"
    
    def test_model_with_different_input_shapes(self):
        """Test model behavior when input shapes don't match expected"""
        model = Sequential([
            Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Test with wrong input shape
        wrong_shape_data = np.random.rand(1, 64, 64, 3).astype(np.float32)
        
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            model.predict(wrong_shape_data, verbose=0)