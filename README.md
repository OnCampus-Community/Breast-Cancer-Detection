# Breast Cancer Detection using Artificial Neural Network
This repository contains code for the detection of breast cancer using an artificial neural network. The model is implemented in Python and makes use of the following libraries:

Pandas

Numpy

Matplotlib

Sklearn

TensorFlow

## Dataset
The dataset used for training and testing the model is a CSV file taken from "https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset" , which contains various attributes related to breast cancer diagnosis. The dataset is pre-processed to obtain relevant information for the model to make predictions.

## Model
The model is implemented using TensorFlow and is a feedforward neural network. It consists of multiple dense layers, with each layer having a certain number of neurons. The model is trained using the backpropagation algorithm, where the model's parameters are updated to minimize the loss function.

## How to run:
Note: This virtual environment created on anaconda command prompt.
#### Step 1: Clone the Repository
Explain how users can clone your project's GitHub repository to their local machine using the git clone command.
```
git clone https://github.com/OnCampus-Community/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```
#### Step 2: Create a Conda Environment
Guide users through the process of creating a Conda environment for your project. Explain why using a Conda environment is beneficial.

``````
conda create --name myenv2 python=X.X
``````
Replace myenv2 with your preferred environment name and X.X with the desired Python version.
#### Step 3: Activate the Conda Environment
Show users how to activate the Conda environment:
``````
conda activate myenv2
``````
#### Step 4: Install Project Dependencies
Explain how to install the project dependencies from the requirements_1.txt file.
The requirement file in this project based on jupyter notebook.
``````
pip install -r requirements_1.txt
``````
#### Step 5: Run the Project
Provide instructions on how to run your project. Include any specific commands or configuration that users need to know.
## Usage
To run the code, simply clone the repository and run the main.py file. The code will load the dataset, pre-process the data, train the model, and make predictions on the test data. You can visualize the model's performance by plotting the accuracy and loss curves.

## Conclusion
The model's accuracy on the test data is quite good and demonstrates the power of artificial neural networks in solving complex problems. This code can be used as a reference for developing more advanced models for breast cancer detection.

Note: The accuracy of the model may vary depending on the dataset used for training. It is recommended to try different neural network architectures and hyperparameters to obtain the best results.

