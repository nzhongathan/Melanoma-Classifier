# Melanoma Classifier
The SIIM-ISIC Melanoma Competition is a computer vision challenged in where medical imaging of skin lesions that hint at melanoma tumors are analyzed and determined whether they are malignant or benign. Training data of skin lesions are used to train computer vision models and are later used to evaluate the test data of more medical images. 

![header (1)](https://user-images.githubusercontent.com/69808907/132391141-6cdc6216-27f3-4994-be63-f3eccc9cc00e.png)

## Importing Libraries
Firstly, I will be importing the necessary libraries to build my model. Although I imported an extensive amount of libraries, the major ones used are:
- NumPy
- Pandas
- cv2
- PIL
- Tensorflow and Keras
- Sklearn

## Preparing the Datasets
I decided to first import the training and testing datasets, which include a directory to the image links in the folders. I next transformed the text data into labels and numbers for the neural network to read and process more easily. I filled in all the NaN and N/A values with arbitrary attributes so all elements in the dataset are numerical values. 

I next created a new dataset where for each kind of skin lesion (head/neck, lower extremity, etc.), I made equal numbers of positive and negative cases. This is because in the dataset, there was an overwhelming number of benign images and not enough malignant, meaning that the neural network would overtrain and favor the benign cases rather than actually testing to see if the tumor is malignant. 

![Benign-and-malignant-melanoma](https://user-images.githubusercontent.com/69808907/132392306-14c2b86d-5f3b-43d6-8813-4fbe068215ed.png)

## Helpful Functions
I created several functions that would be helpful for analyzing the data and training the models later on. These include:
- display_one: Displaying the image of one skin lesion
- display: Plot two images, first one is the original, second is the editted one after preprocessing
- process: Resizing and processing the image to make it usable in Keras models
- no_noise: Preprocessing images with Gaussian Blur (Reduces noise and helps neural network generalize its parameters)
- eroded: Eroding the images
- preprocess: Normalize, run Gaussian Blur, and other preprocessing techniques
- plot: Plotting accuracy and loss curves
- focal_loss: Defining the focal loss function

## Establishing the Neural Network
I will first use the processing and preprocessing functions to prepare our image data for the neural network I will be using. I will split the data into training and testing data in order to train our model. The neural network I will be using is InceptionResNetV2, imported from Keras. I included a Dense layer with a sigmoid activation function to helps turn all the outputs from the CNN into 0s and 1s. 

I set the epoch to 100 with a low learning rate of 0.0001. I also set a decay rate to slow down the learning rate as the epochs continue on, ensuring that the CNN won't overtrain to the training model. I also established a Adam optimizer function and an early stopping function to stop the training when the validation accuracy plateaus. 

![image00](https://user-images.githubusercontent.com/69808907/132393687-26fa4b78-49c5-4fc6-82a9-7ff2af404da4.png)

From there I will utilize my trained CNN and then create a submission file.
