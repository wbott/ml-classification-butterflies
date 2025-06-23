"""Base model class for butterfly classification."""

from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from src.utils.config import config


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, model_name: str):
        """Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.history = None
        
    @abstractmethod
    def build_model(self, input_shape: tuple, num_classes: int) -> Model:
        """Build the model architecture.
        
        Args:
            input_shape: Input shape for the model
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        pass
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     learning_rate: float = None,
                     loss: str = 'categorical_crossentropy',
                     metrics: list = None):
        """Compile the model.
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy']
            
        learning_rate = learning_rate or config.get('training.learning_rate', 1e-4)
        
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
            
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def get_callbacks(self, model_dir: str = None) -> list:
        """Get training callbacks.
        
        Args:
            model_dir: Directory to save model checkpoints
            
        Returns:
            List of callbacks
        """
        model_dir = Path(model_dir) if model_dir else Path(config.get('paths.models_dir'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.get('training.early_stopping_patience', 5),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.get('training.reduce_lr_factor', 0.5),
                patience=config.get('training.reduce_lr_patience', 3),
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(model_dir / f'{self.model_name}_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, 
             train_generator,
             validation_generator,
             epochs: int = None,
             callbacks: list = None,
             class_weight: dict = None) -> dict:
        """Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of epochs to train
            callbacks: List of callbacks
            class_weight: Class weights for imbalanced data
            
        Returns:
            Training history
        """
        epochs = epochs or config.get('training.epochs', 50)
        callbacks = callbacks or self.get_callbacks()
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(self, test_generator) -> dict:
        """Evaluate the model.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
            
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Create dictionary of metric names and values
        metric_names = self.model.metrics_names
        return dict(zip(metric_names, results))
    
    def predict(self, data_generator):
        """Make predictions.
        
        Args:
            data_generator: Data generator for predictions
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
            
        return self.model.predict(data_generator, verbose=1)
    
    def save_model(self, filepath: str):
        """Save the model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built")
            
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built")
        return self.model.summary()