# 🦋 Butterfly Image Classification: A Deep Dive into FNN, CNN, and VGG

Welcome to an exciting journey through the vibrant world of butterfly image classification! This project explores how well we can classify butterfly species using machine learning, pitting three powerful techniques against each other: **Feedforward Neural Networks (FNN)**, **Convolutional Neural Networks (CNN)**, and **Transfer Learning with VGG16**. With a rich dataset and some clever optimizations, we aim to uncover the best approach to identifying these delicate creatures. Let’s flutter into the details!

---

## 🚀 Setup: Get Ready to Soar

To dive into this project, you’ll need a few tools and libraries. Here’s how to set up your environment and automate the process for a smooth takeoff.

### Prerequisites
- **Python 3.8+**: Ensure you have Python installed.
- **VS Code**: A versatile IDE for coding and debugging. Install the Python and Jupyter extensions for a seamless experience.
- **Kaggle API Key**: Required to download the dataset programmatically. Follow these steps:
  1. Sign up/log in to [Kaggle](https://www.kaggle.com).
  2. Go to your account settings, create a new API token, and save the `kaggle.json` file.
  3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\` (Windows).
- **Virtual Environment**: Keep dependencies tidy with a virtual environment.

### Python Libraries
Install the required libraries using pip. Run the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kaggle kagglehub ipykernel jupyterlab_widgets
```

### Automation Script
To streamline setup, here’s a handy Bash script (`setup.sh`) to create a virtual environment, install dependencies, and verify the Kaggle API. Save it, make it executable (`chmod +x setup.sh`), and run it (`./setup.sh`).

```bash
#!/bin/bash
echo "Setting up the Butterfly Classification project..."

# Create and activate virtual environment
python3 -m venv butterfly_env
source butterfly_env/bin/activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kaggle kagglehub ipykernel jupyterlab_widgets

# Verify Kaggle API
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle API key found!"
else
    echo "ERROR: Kaggle API key not found. Please place kaggle.json in ~/.kaggle/"
    exit 1
fi

echo "Setup complete! Activate the environment with 'source butterfly_env/bin/activate'"
```

For Windows, use a similar batch script (`setup.bat`):

```bat
@echo off
echo Setting up the Butterfly Classification project...

:: Create and activate virtual environment
python -m venv butterfly_env
call butterfly_env\Scripts\activate

:: Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kaggle kagglehub ipykernel jupyterlab_widgets

:: Verify Kaggle API
if exist %USERPROFILE%\.kaggle\kaggle.json (
    echo Kaggle API key found!
) else (
    echo ERROR: Kaggle API key not found. Please place kaggle.json in %USERPROFILE%\.kaggle\
    exit /b 1
)

echo Setup complete! Activate the environment with 'butterfly_env\Scripts\activate'
```

### VS Code Tips
- Use the **Jupyter extension** to run the notebook interactively.
- Enable **auto-save** and **format-on-save** for cleaner code.
- Set up a Python interpreter by selecting the virtual environment (`butterfly_env`).

---

## 🌟 General Problem Statement: How Good Can We Get?

Butterflies are nature’s masterpieces, each species flaunting unique patterns and colors. But can a machine learn to distinguish them as well as a seasoned lepidopterist? This project tackles the challenge of **classifying butterfly images into distinct species**, pushing the boundaries of image classification accuracy.

We explore three techniques:
- **Feedforward Neural Networks (FNN)**: A baseline approach, flattening images into vectors for classification.
- **Convolutional Neural Networks (CNN)**: Tailored for images, capturing spatial patterns like edges and textures.
- **Transfer Learning with VGG16**: Leveraging pre-trained models to extract high-level features, fine-tuned for our task.

The problem? Build a model that accurately identifies 75 butterfly species from images, balancing performance and generalization. We’re solving a real-world computer vision challenge with applications in biodiversity monitoring and ecological research.

---

## 📊 The Dataset: A Kaleidoscope of Butterflies

Our dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification), is a treasure trove of butterfly imagery:
- **Size**: Over **9,000 labeled images** in the training set, with additional images for testing.
- **Classes**: **75 unique butterfly species**, each with varying numbers of samples (80–130 images per class).
- **Structure**: 
  - Training images are labeled in `Training_set.csv`.
  - Test images in `Testing_set.csv` require predictions.
  - Images are stored in a `train/` directory, with RGB channels critical for distinguishing mimetic or camouflaged species.

The dataset exhibits a slight class imbalance, with some species having more images than others. This long-tail distribution challenges our models to generalize across all classes effectively.

---

## 🛠️ Our Approach: Three Tactics, One Goal

We trained and compared three models, each with a distinct strategy for tackling butterfly classification. Here’s how we approached each, plus a special tweak to supercharge VGG16.

### 1. Feedforward Neural Network (FNN)
- **What It Does**: Flattens 150x150 RGB images into a 1D vector and processes them through dense layers (1024, 512, 256, 128 neurons) with ReLU activation, batch normalization, and dropout (0.5 to 0.2) to prevent overfitting. The output layer uses softmax for 75-class classification.
- **Why It’s Cool**: A simple baseline, FNN treats images as raw pixel data, testing how far we can get without spatial awareness.
- **Performance**: Achieved ~69% accuracy, decent but limited by its inability to capture spatial patterns.

### 2. Convolutional Neural Network (CNN)
- **What It Does**: Uses three convolutional layers (32, 64, 128 filters) with ReLU activation and max-pooling to extract spatial features, followed by a dense layer (512 neurons) and a softmax output. Trained on 150x150 images.
- **Why It’s Cool**: CNNs are built for images, detecting edges, textures, and shapes, making them a natural fit for this task.
- **Performance**: Shockingly poor at ~1% accuracy, likely due to mismatched input shapes or insufficient model depth. A mystery to investigate!

### 3. Transfer Learning with VGG16
- **What It Does**: Leverages VGG16, pre-trained on ImageNet, with frozen layers to extract features. We add global average pooling, a dense layer (512 neurons), batch normalization, dropout (0.5), and a softmax output for 75 classes. Trained on 224x224 images.
- **Why It’s Cool**: Transfer learning uses pre-trained knowledge, ideal for limited datasets, saving time and boosting accuracy.
- **Performance**: Reached ~76% accuracy, a big leap from FNN, showing the power of pre-trained features.

### 4. Optimized VGG16: The Secret Sauce
- **How We Augmented It**: We refined VGG16 with:
  - **Input Size**: Ensured 224x224 images to match VGG16’s expectations.
  - **Reduced Augmentation**: Lowered rotation, shear, and zoom (to 10–20%) for stability.
  - **Lower Learning Rate**: Used 1e-4 with Adam for smoother convergence.
  - **Callbacks**: Added EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5), and ModelCheckpoint to save the best model.
  - **Batch Normalization**: Enhanced training stability.
  - **Reduced Dropout**: Lowered to 0.3 to balance regularization.
- **Why It Worked**: These tweaks stabilized training, prevented overfitting, and aligned the model with the dataset’s needs.
- **Performance**: Achieved **~83% accuracy** and the lowest loss (~0.65), making it the star performer.

---

## 🎉 Final Takeaway & Next Steps

### Key Insights
- **VGG Optimized Shines**: The fine-tuned VGG16 model delivered the best accuracy (~83%) and loss (~0.65), proving transfer learning’s edge for image classification with limited data.
- **FNN Holds Its Own**: At ~69% accuracy, FNN is a respectable baseline but lacks the spatial prowess of CNNs or VGG.
- **CNN’s Epic Fail**: The ~1% accuracy suggests a critical issue (e.g., input shape mismatch or shallow architecture). It’s a puzzle worth solving!
- **Transfer Learning Wins**: Pre-trained models like VGG16, especially when optimized, outperform custom architectures for this task.

### Next Steps
1. **Debug the CNN**: Investigate input shapes, model summary, and training history to pinpoint why it underperformed.
2. **Fine-Tune VGG**: Unfreeze top VGG16 layers and train with a very low learning rate to squeeze out extra performance.
3. **Expand the Dataset**: Augment with more images or synthetic data to address class imbalance.
4. **Try Other Models**: Experiment with ResNet50 or MobileNet for comparison.
5. **Deploy the Model**: Package `vgg_opt` for real-world use, perhaps as a web app to identify butterflies from user-uploaded images.

This project showcases the thrill of machine learning, from baseline struggles to optimized triumphs. Ready to classify more butterflies or tackle your own image challenge? Fork this repo and spread your wings! 🦋