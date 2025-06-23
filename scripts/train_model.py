#!/usr/bin/env python3
"""Training script for butterfly classification models."""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ButterflyDataset
from src.models.vgg_transfer import VGGTransferModel
from src.utils.config import config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train butterfly classification model')
    parser.add_argument('--model', choices=['vgg16'], default='vgg16',
                       help='Model architecture to use')
    parser.add_argument('--train-csv', required=True,
                       help='Path to training CSV file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for optimizer')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Enable fine-tuning after initial training')
    parser.add_argument('--fine-tune-epochs', type=int, default=10,
                       help='Number of epochs for fine-tuning')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    
    print("🦋 Butterfly Classification Training")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Training CSV: {args.train_csv}")
    print(f"Epochs: {config.get('training.epochs')}")
    print(f"Batch size: {config.get('data.batch_size')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    print()
    
    # Initialize dataset
    dataset = ButterflyDataset()
    
    # Create data generators
    print("📊 Creating data generators...")
    train_gen, val_gen = dataset.create_data_generators(
        train_csv=args.train_csv,
        augment=True
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Number of classes: {train_gen.num_classes}")
    print()
    
    # Initialize model
    if args.model == 'vgg16':
        model = VGGTransferModel()
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Build and compile model
    print("🏗️  Building model...")
    input_shape = tuple(config.get('data.image_size')) + (3,)
    model.build_model(input_shape, train_gen.num_classes)
    model.compile_model()
    
    print("Model built successfully!")
    model.summary()
    print()
    
    # Calculate class weights for imbalanced dataset
    print("⚖️  Calculating class weights...")
    class_weights = dataset.get_class_weights(train_gen)
    print("Class weights calculated.")
    print()
    
    # Train model
    print("🚀 Starting training...")
    history = model.train(
        train_generator=train_gen,
        validation_generator=val_gen,
        class_weight=class_weights
    )
    
    print("✅ Initial training completed!")
    
    # Fine-tuning if requested
    if args.fine_tune:
        print()
        print("🔧 Starting fine-tuning...")
        fine_tune_history = model.fine_tune(
            train_generator=train_gen,
            validation_generator=val_gen,
            epochs=args.fine_tune_epochs
        )
        print("✅ Fine-tuning completed!")
    
    # Save final model
    model_path = Path(config.get('paths.models_dir')) / 'final' / f'{args.model}_final.keras'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"💾 Model saved to: {model_path}")
    
    # Print final metrics
    print()
    print("📈 Final Training Metrics:")
    print("-" * 30)
    final_val_acc = history['val_accuracy'][-1]
    final_val_loss = history['val_loss'][-1]
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")


if __name__ == '__main__':
    main()