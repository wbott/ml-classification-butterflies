"""VGG16 Transfer Learning model for butterfly classification."""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from src.models.base_model import BaseModel
from src.utils.config import config


class VGGTransferModel(BaseModel):
    """VGG16 Transfer Learning model."""
    
    def __init__(self):
        """Initialize VGG transfer learning model."""
        super().__init__('vgg16_transfer')
        
    def build_model(self, input_shape: tuple, num_classes: int) -> Model:
        """Build VGG16 transfer learning model.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Dense layer
        dense_units = config.get('models.vgg16.dense_units', 512)
        x = Dense(dense_units, activation='relu', name='dense_512')(x)
        
        # Batch normalization if configured
        if config.get('models.vgg16.batch_norm', True):
            x = BatchNormalization(name='batch_norm')(x)
        
        # Dropout
        dropout_rate = config.get('models.vgg16.dropout_rate', 0.3)
        x = Dropout(dropout_rate, name='dropout')(x)
        
        # Output layer
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        return self.model
    
    def unfreeze_top_layers(self, num_layers: int = 4):
        """Unfreeze top layers for fine-tuning.
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Find VGG16 layers (they start with 'block' prefix in the flattened model)
        vgg_layers = []
        for layer in self.model.layers:
            if layer.name.startswith('block'):  # VGG16 layers
                vgg_layers.append(layer)
        
        if len(vgg_layers) == 0:
            raise ValueError("Could not find VGG16 layers in the model")
        
        # Unfreeze top layers (last num_layers of VGG16)
        layers_to_unfreeze = vgg_layers[-num_layers:]
        for layer in layers_to_unfreeze:
            layer.trainable = True
        
        print(f"Unfroze top {num_layers} VGG16 layers: {[l.name for l in layers_to_unfreeze]}")
        
        # Recompile with lower learning rate for fine-tuning
        self.compile_model(learning_rate=1e-5)
    
    def fine_tune(self,
                  train_generator,
                  validation_generator,
                  epochs: int = 10,
                  num_layers_to_unfreeze: int = 4):
        """Fine-tune the model by unfreezing top layers.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of epochs for fine-tuning
            num_layers_to_unfreeze: Number of top layers to unfreeze
            
        Returns:
            Fine-tuning history
        """
        # Unfreeze top layers
        self.unfreeze_top_layers(num_layers_to_unfreeze)
        
        # Fine-tune with fewer epochs and lower learning rate
        fine_tune_history = self.train(
            train_generator=train_generator,
            validation_generator=validation_generator,
            epochs=epochs
        )
        
        return fine_tune_history
    
    def get_feature_extractor(self):
        """Get feature extractor (base model without classification head).
        
        Returns:
            Feature extractor model
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Find the global average pooling layer (name includes suffix)
        gap_layer = None
        for layer in self.model.layers:
            if 'global_average_pooling2d' in layer.name:
                gap_layer = layer
                break
        
        if gap_layer is None:
            raise ValueError("Could not find GlobalAveragePooling2D layer")
        
        # Create feature extractor up to global average pooling
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=gap_layer.output
        )
        
        return feature_extractor