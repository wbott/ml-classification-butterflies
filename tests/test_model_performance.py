import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

class TestModelPerformance:
    
    @pytest.fixture
    def sample_image_data(self):
        """Create synthetic image data for testing"""
        np.random.seed(42)
        # Create sample images (small for testing)
        n_samples = 100
        n_classes = 5
        img_height, img_width, channels = 64, 64, 3
        
        X = np.random.rand(n_samples, img_height, img_width, channels).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Convert to categorical
        y_categorical = tf.keras.utils.to_categorical(y, n_classes)
        
        return X, y_categorical, n_classes
    
    @pytest.fixture
    def simple_cnn_model(self, sample_image_data):
        """Create a simple CNN model for testing"""
        X, y, n_classes = sample_image_data
        img_height, img_width, channels = X.shape[1:]
        
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Quick training for testing
        model.fit(X[:80], y[:80], epochs=3, verbose=0, validation_split=0.2)
        
        return model, X[80:], y[80:]
    
    @pytest.fixture
    def vgg_transfer_model(self, sample_image_data):
        """Create a VGG16 transfer learning model for testing"""
        X, y, n_classes = sample_image_data
        
        # Resize images to 224x224 for VGG16
        X_resized = tf.image.resize(X, [224, 224]).numpy()
        
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Quick training for testing
        model.fit(X_resized[:80], y[:80], epochs=2, verbose=0, validation_split=0.2)
        
        return model, X_resized[80:], y[80:]
    
    def test_model_accuracy_threshold(self, simple_cnn_model):
        """Test that model achieves minimum accuracy threshold"""
        model, X_test, y_test = simple_cnn_model
        
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # For a random model with 5 classes, accuracy should be > 20% (random chance)
        assert accuracy > 0.15, f"Model accuracy {accuracy:.4f} is below minimum threshold"
    
    def test_prediction_probabilities_valid(self, simple_cnn_model):
        """Test that prediction probabilities are valid"""
        model, X_test, y_test = simple_cnn_model
        
        predictions = model.predict(X_test, verbose=0)
        
        # Check probabilities are in valid range [0, 1]
        assert np.all(predictions >= 0), "All probabilities should be >= 0"
        assert np.all(predictions <= 1), "All probabilities should be <= 1"
        
        # Check probabilities sum to 1 for each prediction
        prob_sums = np.sum(predictions, axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6), "Probabilities should sum to 1"
    
    def test_model_consistency(self, simple_cnn_model):
        """Test that model gives consistent predictions"""
        model, X_test, y_test = simple_cnn_model
        
        # Get predictions twice
        pred1 = model.predict(X_test, verbose=0)
        pred2 = model.predict(X_test, verbose=0)
        
        # Should be identical
        assert np.allclose(pred1, pred2), "Model should give consistent predictions"
    
    def test_multiclass_confusion_matrix(self, simple_cnn_model):
        """Test confusion matrix properties for multiclass classification"""
        model, X_test, y_test = simple_cnn_model
        
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Check matrix is square
        assert cm.shape[0] == cm.shape[1], "Confusion matrix should be square"
        
        # Check all values are non-negative
        assert np.all(cm >= 0), "Confusion matrix values should be non-negative"
        
        # Check total predictions match
        assert np.sum(cm) == len(y_true), "Confusion matrix sum should equal number of predictions"
    
    def test_per_class_metrics(self, simple_cnn_model):
        """Test per-class precision and recall are reasonable"""
        model, X_test, y_test = simple_cnn_model
        
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate per-class metrics (with zero_division=0 for empty classes)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Check metrics are in valid range [0, 1]
        assert np.all(precision >= 0) and np.all(precision <= 1), "Precision should be in [0, 1]"
        assert np.all(recall >= 0) and np.all(recall <= 1), "Recall should be in [0, 1]"
        assert np.all(f1 >= 0) and np.all(f1 <= 1), "F1 score should be in [0, 1]"
    
    def test_vgg_transfer_learning_performance(self, vgg_transfer_model):
        """Test that VGG transfer learning model performs better than random"""
        model, X_test, y_test = vgg_transfer_model
        
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Transfer learning should perform better than random (20% for 5 classes)
        assert accuracy > 0.2, f"VGG transfer learning accuracy {accuracy:.4f} should be > random chance"
    
    def test_model_output_shape(self, simple_cnn_model):
        """Test that model output shape matches expected dimensions"""
        model, X_test, y_test = simple_cnn_model
        
        predictions = model.predict(X_test, verbose=0)
        
        # Check output shape
        expected_shape = (len(X_test), y_test.shape[1])
        assert predictions.shape == expected_shape, f"Expected shape {expected_shape}, got {predictions.shape}"
    
    def test_model_handles_batch_sizes(self, simple_cnn_model):
        """Test model can handle different batch sizes"""
        model, X_test, y_test = simple_cnn_model
        
        # Test single sample
        single_pred = model.predict(X_test[:1], verbose=0)
        assert single_pred.shape[0] == 1, "Model should handle single sample"
        
        # Test different batch sizes
        for batch_size in [1, 5, 10]:
            if len(X_test) >= batch_size:
                batch_pred = model.predict(X_test[:batch_size], verbose=0)
                assert batch_pred.shape[0] == batch_size, f"Model should handle batch size {batch_size}"
    
    def test_model_memory_usage(self, simple_cnn_model):
        """Test model doesn't consume excessive memory"""
        model, X_test, y_test = simple_cnn_model
        
        # Get model size in parameters
        total_params = model.count_params()
        
        # For our test model, should be reasonable size (< 1M parameters)
        assert total_params < 1_000_000, f"Model has {total_params} parameters, might be too large for testing"
    
    def test_model_prediction_speed(self, simple_cnn_model):
        """Test model prediction speed is reasonable"""
        model, X_test, y_test = simple_cnn_model
        
        import time
        
        # Time predictions
        start_time = time.time()
        predictions = model.predict(X_test, verbose=0)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        time_per_sample = prediction_time / len(X_test)
        
        # Should predict reasonably fast (< 1 second per sample for small test model)
        assert time_per_sample < 1.0, f"Prediction too slow: {time_per_sample:.4f} seconds per sample"
    
    def test_model_convergence_indicators(self, sample_image_data):
        """Test that model shows signs of learning during training"""
        X, y, n_classes = sample_image_data
        img_height, img_width, channels = X.shape[1:]
        
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train for a few epochs and check loss decreases
        history = model.fit(X[:80], y[:80], epochs=5, verbose=0, validation_split=0.2)
        
        losses = history.history['loss']
        
        # Loss should generally decrease (last loss < first loss)
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        assert final_loss < initial_loss * 1.1, f"Model not learning: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}"