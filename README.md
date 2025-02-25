**Bird vs Hare Image Classification using Convolutional Neural Networks (CNN)**
===============================================================================

📜 **Overview**
---------------

Welcome to the **Bird vs Hare Image Classification** project! This repository demonstrates how to build a **Convolutional Neural Network (CNN)** model using **TensorFlow** and **Keras** to classify images of **bald eagles (birds)** and **hares**. The model is trained and evaluated on a dataset of images, and it achieves impressive accuracy for distinguishing between these two classes.

This project is ideal for anyone looking to understand how to apply deep learning models to image classification tasks and get familiar with working with CNNs.

🛠 **Technologies Used**
------------------------

*   **TensorFlow & Keras**: For building and training the CNN model.
*   **NumPy**: For data manipulation.
*   **Matplotlib**: For visualizing training history and images.
*   **Pillow**: For image preprocessing.
*   **OpenCV**: For image display and manipulation.
*   **scikit-learn**: For splitting the dataset into training and test sets.

📊 **Dataset**
--------------

The dataset contains two classes of images:

1.  **Bald Eagles (Birds)**
2.  **Hares**

Each image is resized to **128x128** pixels and is converted to **RGB format** for training the CNN.

*   **Bald Eagles** images are stored under `/content/data(CNN)/bird bald_eagle/`
*   **Hares** images are stored under `/content/data(CNN)/hare/`

### **Dataset Summary**

**Class**

**Number of Images**

**Bald Eagles**

1460

**Hares**

1356

**Total**

2816

### **Sample Images**

1.  **Bald Eagle**  
    
2.  **Hare**  
    

⚙️ **Model Architecture**
-------------------------

The CNN model built for this project has the following architecture:

*   **Input Layer**: (128x128x3) - RGB Image
*   **Conv2D Layer**: 32 filters, (3x3) kernel, ReLU activation
*   **MaxPooling2D Layer**: (2x2) pool size
*   **Conv2D Layer**: 64 filters, (3x3) kernel, ReLU activation
*   **MaxPooling2D Layer**: (2x2) pool size
*   **Flatten Layer**
*   **Dense Layer**: 128 neurons, ReLU activation
*   **Dropout Layer**: 50% dropout
*   **Dense Layer**: 64 neurons, ReLU activation
*   **Dropout Layer**: 50% dropout
*   **Output Layer**: 2 neurons (softmax activation for binary classification)

### **Optimizer and Loss Function**

*   **Optimizer**: Adam
*   **Loss Function**: Sparse Categorical Crossentropy

🚀 **Training Results**
-----------------------

The model was trained for **10 epochs** with an 80-20 training-test split.

**Metric**

**Train Value**

**Validation Value**

**Loss**

0.4191

0.4207

**Accuracy**

86.64%

86.52%

As you can see, the model achieves excellent performance on both the training and validation datasets.

📈 **Training Progress**
------------------------

### **Loss Graph**

The loss graph shows the reduction in error as the model learns over time:

### **Accuracy Graph**

The accuracy graph shows how the model's prediction accuracy improves with each epoch:

🔍 **Model Evaluation**
-----------------------

After training, we evaluated the model on the test dataset, and the following results were obtained:

*   **Test Loss**: 0.4207
*   **Test Accuracy**: 86.52%

### **Sample Prediction Results**

Here are the results of the model's predictions:

1.  **Prediction for a Hare Image**
    
    *   **Model Prediction**: Hare
    *   **Actual Image**:
        
2.  **Prediction for a Bird Image**
    
    *   **Model Prediction**: Bird
    *   **Actual Image**:
        

🔧 **How to Use the Code**
--------------------------

### 1\. **Clone the Repository**

Start by cloning this repository to your local machine:

`git clone https://github.com/your-username/bird-vs-hare-image-classification.git
cd bird-vs-hare-image-classification` 

### 2\. **Install the Required Libraries**

You will need Python 3.x and the following libraries:

`pip install tensorflow numpy pandas matplotlib opencv-python pillow scikit-learn` 

### 3\. **Prepare the Dataset**

The dataset is contained within a **zip file**. Make sure to unzip it in the correct directory. If the dataset is located at `/content/data(CNN).zip`, you can extract it as follows:

`from zipfile import ZipFile
with ZipFile('data(CNN).zip', 'r') as zip_ref:
    zip_ref.extractall('data(CNN)')` 

### 4\. **Run the Model Training**

Run the script to start training:

`python train_model.py` 

### 5\. **Make Predictions**

After training, you can input new images for classification. Use the following code to predict whether an image is of a hare or a bald eagle:

`image_path = 'path_to_image.jpg'
prediction = model.predict(image_path)
print(prediction)` 

🛠 **Future Improvements**
--------------------------

*   **Data Augmentation**: Add techniques like flipping, rotation, and scaling to augment the dataset, making the model more robust.
*   **Hyperparameter Optimization**: Tune hyperparameters like the learning rate, batch size, and network architecture to improve accuracy further.
*   **Transfer Learning**: Experiment with pre-trained models like VGG16 or ResNet50 to leverage existing knowledge for better performance.

* * *

### **Conclusion**

This project demonstrates how to successfully build a deep learning model for image classification tasks using CNNs. With solid accuracy and a well-structured architecture, this model can be adapted to other classification problems with similar image datasets.

We hope you find this project useful, and feel free to contribute or improve it!