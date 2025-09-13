# Facial Expression Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify human facial expressions from images. The model is built using TensorFlow and Keras and is trained to recognize seven different emotions: **angry, disgust, fear, happy, neutral, sad, and surprise**.

## 📝 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## 🧐 Project Overview
The goal of this project is to build an effective deep learning model for emotion detection from facial images. The process involves loading and preprocessing the image data, performing exploratory data analysis (EDA), building a custom CNN architecture, training the model, and visualizing its performance.

## 🖼️ Dataset
- The model is trained on the Facial Expression Dataset, which consists of 48x48 pixel grayscale images.
- **Training Set:** Contains 28,709 images.
- **Testing Set:** Contains 7,178 images.
- The images are categorized into seven distinct emotion classes. The exploratory data analysis revealed an imbalance in the dataset, with the 'happy' class being the most represented and the 'disgust' class being the least.
- **Download**: [Dataset (Google Drive)](https://drive.google.com/file/d/134M73bJhOZRIiyYGx4VSoRldIFlz_Zix/view?usp=sharing) 

*Visualization of the class distribution in the training data:*

![Class Distribution Visualization](https://via.placeholder.com/600x400?text=Class+Distribution+Visualization)

## ⚙️ Methodology
The project follows a standard machine learning workflow:

1. **Data Loading:** Image paths and corresponding labels for both training and testing sets are loaded into pandas DataFrames. The training data is shuffled to ensure randomness.
2. **Exploratory Data Analysis (EDA):** The class distribution is visualized using a count plot to understand the data imbalance. Sample images are displayed to inspect their properties (e.g., resolution, grayscale format).
3. **Feature Extraction & Preprocessing:**
    - Images are loaded and converted into 48x48 grayscale NumPy arrays.
    - The pixel values are normalized from the range [0, 255] to [0, 1] to aid in model training.
    - The categorical string labels (e.g., 'happy') are converted into numerical format using LabelEncoder and then transformed into one-hot encoded vectors.
4. **Model Training:** The CNN model is trained for 100 epochs with a batch size of 128. The training progress, including accuracy and loss, is monitored on both the training and validation sets.

## 🧠 Model Architecture
A Sequential CNN model was created with the following structure:

- **Four Convolutional Blocks:** Each block consists of:
    - A Conv2D layer with relu activation. The number of filters increases with depth (128, 256, 512, 512).
    - A MaxPooling2D layer to downsample the feature maps.
    - A Dropout layer (with a rate of 0.4) to prevent overfitting.
- **Flatten Layer:** To convert the 2D feature maps into a 1D vector.
- **Two Fully Connected (Dense) Layers:**
    - A Dense layer with 512 units and relu activation, followed by Dropout.
    - A Dense layer with 256 units and relu activation, followed by Dropout.
- **Output Layer:** A Dense layer with 7 units (one for each class) and a softmax activation function for multi-class classification.

The model is compiled using the Adam optimizer and categorical_crossentropy as the loss function.

```python
model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 📈 Results
After 100 epochs, the model achieved:
- **Training Accuracy:** ~71.5%
- **Validation Accuracy:** ~63.7%

The training history plots below show the model's accuracy and loss over the epochs. There is a noticeable gap between the training and validation curves, which suggests the model may be slightly overfitting to the training data.

| Accuracy Plot | Loss Plot |
|--------------|-----------|
| <img src="https://github.com/user-attachments/assets/d37d7bca-5afc-4f41-898a-4887547f160b" width="400"/> | <img src="https://github.com/user-attachments/assets/db4392e2-bfd0-428a-a8a6-c8a00b77db67" width="400"/> |

## 🛠️ Technologies Used
- Python 3
- TensorFlow & Keras: For building and training the deep learning model.
- Pandas: For data manipulation and management.
- NumPy: For numerical operations.
- Matplotlib & Seaborn: For data visualization.
- Scikit-learn: For data preprocessing (LabelEncoder).
- Pillow (PIL): For image processing.
- Jupyter Notebook: As the development environment.

## 🚀 How to Run
To run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adarshukla3005/Expression-Recognition-Analysis.git
   cd facial-expression-recognition
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   > **Note:** If `requirements.txt` does not exist, create it with the libraries listed below.

   Example `requirements.txt`:
   ```
   tensorflow
   keras
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   pillow
   jupyter
   ```

3. **Download the Dataset:**
   - Download the dataset from a source like Kaggle.
   - Unzip the files and place them in the following directory structure (relative to your project root):

   ```
   dataset/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```
   - Alternatively, update the `TRAIN_DIR` and `TEST_DIR` variables in the notebook to point to your dataset location.

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   - Open the notebook file (`.ipynb`) and run the cells sequentially.

---

**If you encounter any issues or have questions, please open an issue or contact the maintainer.**
