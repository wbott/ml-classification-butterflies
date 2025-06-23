"""Dataset handling utilities for butterfly classification."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from src.utils.config import config


class ButterflyDataset:
    """Butterfly dataset handler."""
    
    def __init__(self, data_dir: str = None):
        """Initialize dataset handler.
        
        Args:
            data_dir: Directory containing the dataset
        """
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('paths.data_dir'))
        self.label_encoder = LabelEncoder()
        self.num_classes = config.get('data.num_classes', 75)
        
    def load_metadata(self, csv_path: str) -> pd.DataFrame:
        """Load dataset metadata from CSV.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with image filenames and labels
        """
        df = pd.read_csv(csv_path)
        return df
    
    def create_data_generators(self, 
                             train_csv: str,
                             validation_split: float = None,
                             batch_size: int = None,
                             image_size: Tuple[int, int] = None,
                             augment: bool = True) -> Tuple:
        """Create data generators for training and validation.
        
        Args:
            train_csv: Path to training CSV file
            validation_split: Fraction of data for validation
            batch_size: Batch size for generators
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        validation_split = validation_split or config.get('data.validation_split', 0.2)
        batch_size = batch_size or config.get('data.batch_size', 32)
        image_size = image_size or config.get('data.image_size', [224, 224])
        
        # Load metadata
        df = self.load_metadata(train_csv)
        
        # Prepare data generators
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            df,
            directory=self.data_dir / 'raw' / 'train',
            x_col='filename',
            y_col='label',
            target_size=tuple(image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = validation_datagen.flow_from_dataframe(
            df,
            directory=self.data_dir / 'raw' / 'train',
            x_col='filename',
            y_col='label',
            target_size=tuple(image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def create_test_generator(self,
                            test_csv: str,
                            batch_size: int = None,
                            image_size: Tuple[int, int] = None):
        """Create test data generator.
        
        Args:
            test_csv: Path to test CSV file
            batch_size: Batch size for generator
            image_size: Target image size (height, width)
            
        Returns:
            Test data generator
        """
        batch_size = batch_size or config.get('data.batch_size', 32)
        image_size = image_size or config.get('data.image_size', [224, 224])
        
        df = self.load_metadata(test_csv)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_dataframe(
            df,
            directory=self.data_dir / 'raw' / 'test',
            x_col='filename',
            y_col=None,
            target_size=tuple(image_size),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False
        )
        
        return test_generator
    
    def get_class_names(self, generator) -> list:
        """Get class names from generator.
        
        Args:
            generator: Data generator
            
        Returns:
            List of class names
        """
        return list(generator.class_indices.keys())
    
    def get_class_weights(self, generator) -> dict:
        """Calculate class weights for imbalanced dataset.
        
        Args:
            generator: Training data generator
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get labels from generator
        labels = generator.classes
        unique_labels = np.unique(labels)
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        return dict(zip(unique_labels, class_weights))