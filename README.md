# CNN Image Classifier

This project implements a powerful image classification system using deep learning models. The system is designed to analyze and categorize input images, featuring an interactive web interface for a seamless user experience. It offers two distinct output modes: a "Simple Prediction" for quick results and a "Detailed Analysis" for in-depth insights.

---

## 🌟 Features

- **Accurate Image Classification**: Utilizes Convolutional Neural Networks (CNNs) for high-accuracy predictions.
- **Interactive User Interface**: A user-friendly web application built with Gradio.
- **Dual Output Modes**:
    - **Simple Prediction Mode**: Displays the final classification result and its confidence score.
    - **Detailed Analysis Mode**: Provides a complete breakdown of predictions, including a probability distribution chart for each class.
- **Automatic Preprocessing**: Input images are automatically resized and normalized to meet the model's requirements.
- **Modular Codebase**: The code is organized into two main components (model creation and UI), making it easy to understand and extend.

---

## 🚀 Project Structure

```
image-classifier/
│
├── create_model.ipynb          # Notebook for building, training, and saving the CNN model
├── ui.ipynb                    # Notebook for running the Gradio user interface
│
├── apple_orange_classifier.h5  # The trained model file (output of create_model.ipynb)
│
├── requirements.txt            # List of required libraries and dependencies
└── README.md                   # Project description and instructions
```
---

## 📋 Requirements

You will need the following libraries to run this project. The full list is available in the `requirements.txt` file.

### Core Dependencies
- **TensorFlow** (for model creation and training)
- **Gradio** (for the web interface)
- **NumPy** (for numerical operations and array handling)
- **Pillow** (for image processing)
- **Matplotlib** (for plotting charts in the detailed analysis)

---

## 🛠️ Installation

1.  **Clone the Repository:**
    First, clone the project from the Git repository:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all the required libraries using the following command:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 How to Use

The project runs in two main steps:

### Step 1: Build and Train the Model

- Run the `create_model.ipynb` notebook.
- This script will build the neural network model, train it on the dataset, and save the final trained model as `apple_orange_classifier.h5`.

**Note:** This step only needs to be performed once to create the model file.

### Step 2: Run the User Interface

- Run the `ui.ipynb` notebook.
- This script will load the saved model (`apple_orange_classifier.h5`) and launch the Gradio user interface.
- After running, a Public URL will be displayed. You can open this link in your web browser to interact with the application.

---

## 🎯 Using the Interface

1.  **Launch the UI:**
    After running `ui.ipynb`, you will get a link similar to this:
    ```
    Running on public URL: https://xxxxxxxxx.gradio.live
    ```

2.  **Upload an Image:**
    In the web interface, upload an image from your local system.

3.  **Get the Result:**
    - In the **"Simple Prediction"** tab, click the "Classify Image" button to see the classification result and confidence score.
    - In the **"Detailed Analysis"** tab, click the "Analyze Image" button to view a detailed breakdown, including a probability distribution plot for each class.

---

## 🔧 Configurable Parameters

You can modify the following parameters in the `create_model.ipynb` notebook to customize the model training process:

python
# In the data loading section
image_size = (32, 32)
batch_size = 32

# In the model compilation section
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# In the model training section
epochs = 10

---

## 💡 Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

1.  **Input Layer**: Accepts color images of size `(32, 32, 3)`.
2.  **Convolutional & Pooling Layers**: Multiple `Conv2D` layers with `ReLU` activation functions to extract features. Each is followed by a `MaxPooling2D` layer to reduce dimensionality.
3.  **Dropout Layers**: Included to prevent overfitting and improve model generalization.
4.  **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
5.  **Fully Connected (Dense) Layers**: Several `Dense` layers to learn high-level patterns from the extracted features.
6.  **Output Layer**: A single neuron with a `Sigmoid` activation function, which outputs a probability between 0 and 1 for binary classification.
`

---
