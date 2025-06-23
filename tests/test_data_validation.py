import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import tempfile
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

class TestDataValidation:
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data mimicking butterfly dataset structure"""
        data = {
            'filename': [f'Image_{i}.jpg' for i in range(1, 101)],
            'label': np.random.choice([
                'MONARCH', 'SWALLOWTAIL', 'BLUE MORPHO', 'CABBAGE WHITE', 'PAINTED LADY'
            ], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_image_directory(self):
        """Create temporary directory with sample images"""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample images
        for i in range(1, 11):
            img = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
            img_pil = Image.fromarray(img)
            img_path = os.path.join(temp_dir, f'Image_{i}.jpg')
            img_pil.save(img_path)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_csv_file_structure(self, sample_csv_data):
        """Test CSV file has required columns and structure"""
        df = sample_csv_data
        
        # Check required columns exist
        required_columns = ['filename', 'label']
        missing_columns = set(required_columns) - set(df.columns)
        assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"
        
        # Check no empty dataframe
        assert len(df) > 0, "CSV file should not be empty"
        
        # Check no null values in critical columns
        assert not df['filename'].isnull().any(), "Filename column should not have null values"
        assert not df['label'].isnull().any(), "Label column should not have null values"
    
    def test_filename_format(self, sample_csv_data):
        """Test image filenames have correct format"""
        df = sample_csv_data
        
        # Check all filenames end with image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in df['filename']:
            has_valid_ext = any(filename.lower().endswith(ext) for ext in valid_extensions)
            assert has_valid_ext, f"Filename {filename} doesn't have valid image extension"
    
    def test_class_distribution(self, sample_csv_data):
        """Test class distribution is reasonable"""
        df = sample_csv_data
        
        class_counts = df['label'].value_counts()
        
        # Check we have multiple classes
        assert len(class_counts) > 1, "Dataset should have multiple classes"
        
        # Check no class is completely dominant (>80% of data)
        max_class_percentage = class_counts.max() / len(df) * 100
        assert max_class_percentage < 80, f"Single class represents {max_class_percentage:.1f}% of data (too dominant)"
        
        # Check no class is extremely rare (<1% of data for datasets >100 samples)
        if len(df) > 100:
            min_class_percentage = class_counts.min() / len(df) * 100
            assert min_class_percentage > 1, f"Smallest class represents only {min_class_percentage:.1f}% of data (too rare)"
    
    def test_no_duplicate_filenames(self, sample_csv_data):
        """Test that there are no duplicate filenames"""
        df = sample_csv_data
        
        duplicate_count = df['filename'].duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate filenames"
    
    def test_image_files_exist(self, sample_csv_data, sample_image_directory):
        """Test that image files referenced in CSV actually exist"""
        df = sample_csv_data.head(10)  # Test first 10 for efficiency
        image_dir = sample_image_directory
        
        missing_files = []
        for filename in df['filename']:
            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                missing_files.append(filename)
        
        assert len(missing_files) == 0, f"Missing image files: {missing_files}"
    
    def test_image_loading_validation(self, sample_image_directory):
        """Test that images can be loaded properly"""
        image_files = [f for f in os.listdir(sample_image_directory) if f.endswith('.jpg')]
        
        for img_file in image_files[:5]:  # Test first 5 images
            img_path = os.path.join(sample_image_directory, img_file)
            
            # Test loading with PIL/Keras
            try:
                img = load_img(img_path)
                img_array = img_to_array(img)
                assert img_array is not None, f"Failed to load image {img_file}"
                assert len(img_array.shape) == 3, f"Image {img_file} should have 3 dimensions (H, W, C)"
            except Exception as e:
                pytest.fail(f"Failed to load image {img_file}: {str(e)}")
    
    def test_image_dimensions_consistency(self, sample_image_directory):
        """Test that images have consistent or valid dimensions"""
        image_files = [f for f in os.listdir(sample_image_directory) if f.endswith('.jpg')]
        
        dimensions = []
        for img_file in image_files:
            img_path = os.path.join(sample_image_directory, img_file)
            img = load_img(img_path)
            dimensions.append(img.size)  # (width, height)
        
        # Check all images have reasonable dimensions
        for dim in dimensions:
            width, height = dim
            assert width > 0 and height > 0, f"Invalid image dimensions: {dim}"
            assert width >= 32 and height >= 32, f"Image too small: {dim}"
            assert width <= 5000 and height <= 5000, f"Image too large: {dim}"
    
    def test_image_color_channels(self, sample_image_directory):
        """Test that images have correct number of color channels"""
        image_files = [f for f in os.listdir(sample_image_directory) if f.endswith('.jpg')]
        
        for img_file in image_files[:3]:  # Test first 3 images
            img_path = os.path.join(sample_image_directory, img_file)
            img = load_img(img_path)
            img_array = img_to_array(img)
            
            # Should be RGB (3 channels)
            assert img_array.shape[2] == 3, f"Image {img_file} should have 3 color channels, got {img_array.shape[2]}"
    
    def test_image_data_generator_creation(self, sample_csv_data, sample_image_directory):
        """Test ImageDataGenerator can be created with the dataset"""
        df = sample_csv_data.head(10)
        image_dir = sample_image_directory
        
        # Create basic data generator
        datagen = ImageDataGenerator(rescale=1./255)
        
        try:
            generator = datagen.flow_from_dataframe(
                dataframe=df,
                directory=image_dir,
                x_col='filename',
                y_col='label',
                target_size=(64, 64),  # Small size for testing
                batch_size=2,
                class_mode='categorical',
                shuffle=False  # For consistent testing
            )
            
            # Test we can get a batch
            batch_x, batch_y = next(generator)
            
            # Validate batch structure
            assert batch_x.shape[0] <= 2, "Batch size should be <= 2"
            assert batch_x.shape[1:] == (64, 64, 3), "Batch images should be (64, 64, 3)"
            assert len(batch_y.shape) == 2, "Batch labels should be 2D (one-hot encoded)"
            
        except Exception as e:
            pytest.fail(f"Failed to create data generator: {str(e)}")
    
    def test_image_augmentation_bounds(self):
        """Test that image augmentation parameters are within reasonable bounds"""
        # Test various augmentation parameters
        augmentation_configs = [
            {'rotation_range': 40, 'width_shift_range': 0.2, 'height_shift_range': 0.2},
            {'shear_range': 0.2, 'zoom_range': 0.2, 'horizontal_flip': True},
            {'brightness_range': [0.8, 1.2]}
        ]
        
        for config in augmentation_configs:
            try:
                datagen = ImageDataGenerator(rescale=1./255, **config)
                assert datagen is not None, f"Failed to create ImageDataGenerator with config: {config}"
            except Exception as e:
                pytest.fail(f"Invalid augmentation config {config}: {str(e)}")
    
    def test_label_encoding_consistency(self, sample_csv_data):
        """Test that labels can be consistently encoded"""
        df = sample_csv_data
        
        unique_labels = df['label'].unique()
        
        # Check we have reasonable number of classes
        assert len(unique_labels) >= 2, "Should have at least 2 classes"
        assert len(unique_labels) <= 1000, "Should not have more than 1000 classes (likely error)"
        
        # Check labels are strings
        for label in unique_labels:
            assert isinstance(label, str), f"Label {label} should be string, got {type(label)}"
            assert len(label) > 0, "Labels should not be empty strings"
    
    def test_train_test_split_proportions(self, sample_csv_data):
        """Test that train/test splits maintain reasonable proportions"""
        df = sample_csv_data
        
        from sklearn.model_selection import train_test_split
        
        # Test different split ratios
        for test_size in [0.2, 0.3]:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            
            actual_test_ratio = len(test_df) / len(df)
            expected_ratio = test_size
            
            # Allow small deviation due to rounding
            assert abs(actual_test_ratio - expected_ratio) < 0.05, \
                f"Test split ratio {actual_test_ratio:.3f} deviates too much from expected {expected_ratio}"
    
    def test_image_pixel_value_ranges(self, sample_image_directory):
        """Test that image pixel values are in expected ranges"""
        image_files = [f for f in os.listdir(sample_image_directory) if f.endswith('.jpg')]
        
        for img_file in image_files[:3]:  # Test first 3 images
            img_path = os.path.join(sample_image_directory, img_file)
            img = load_img(img_path)
            img_array = img_to_array(img)
            
            # Before normalization, pixels should be in [0, 255]
            assert img_array.min() >= 0, f"Image {img_file} has negative pixel values"
            assert img_array.max() <= 255, f"Image {img_file} has pixel values > 255"
            
            # After normalization
            normalized = img_array / 255.0
            assert normalized.min() >= 0, f"Normalized image {img_file} has negative values"
            assert normalized.max() <= 1, f"Normalized image {img_file} has values > 1"
    
    def test_class_mapping_consistency(self, sample_csv_data, sample_image_directory):
        """Test that class mappings are consistent in data generators"""
        df = sample_csv_data.head(10)
        image_dir = sample_image_directory
        
        datagen = ImageDataGenerator(rescale=1./255)
        
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col='filename',
            y_col='label',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Check class indices are consistent
        class_indices = generator.class_indices
        
        assert isinstance(class_indices, dict), "Class indices should be a dictionary"
        assert len(class_indices) > 0, "Should have at least one class"
        
        # Check all values are integers and sequential
        indices = list(class_indices.values())
        assert all(isinstance(i, int) for i in indices), "Class indices should be integers"
        assert min(indices) == 0, "Class indices should start from 0"
        assert max(indices) == len(indices) - 1, "Class indices should be sequential"
    
    def test_batch_size_handling(self, sample_csv_data, sample_image_directory):
        """Test data generator handles different batch sizes correctly"""
        df = sample_csv_data.head(10)
        image_dir = sample_image_directory
        
        datagen = ImageDataGenerator(rescale=1./255)
        
        for batch_size in [1, 2, 5]:
            generator = datagen.flow_from_dataframe(
                dataframe=df,
                directory=image_dir,
                x_col='filename',
                y_col='label',
                target_size=(64, 64),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            batch_x, batch_y = next(generator)
            
            # Batch size should be <= requested size (might be smaller for last batch)
            assert batch_x.shape[0] <= batch_size, f"Batch size {batch_x.shape[0]} exceeds requested {batch_size}"
            assert batch_x.shape[0] > 0, "Batch should not be empty"