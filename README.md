# PhotoOCR-Devanagari_Script_Recognition
This project involves the creation and training of a Neural Network to recognize the Devanagari (Sanskrit) Digits and Characters.

## Description
Created and trained a multiclass classifier Neural Network that can classify the handwritten Devanagari digits (numbers) and characters (letters) given in the form of images of 32x32 pixels with grayscale pixel intensities.

## Modules Used
* ***matplotlib*** to read the images and to plot graphs with obtained data.
* ***os*** and ***sys*** to perform operations on file system to label images and to take arguments.
* ***pandas*** to write and read images with help of dataframes to and from csv files.
* ***numpy*** to perform numerical computations on the data.
* ***scipy*** to perform scientific computations (such as activation functions) on data.

## Methodology
### 1. Gathering the Dataset
The dataset consisting of 92,000 images divided into training(85%) and validation(15%) subfolders is downloaded from [**University of California - Irvine : Machine Learning Repository**](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset).<br/>
The training dataset consists of 46 (36 + 10) subfolders for each character (letter) and digit (number), with each sub-folder containing 1,700 images.<br/>
Similarly, the validation dataset consists of 46 (36 + 10) subfolders for each character (letter) and digit (number), with each sub-folder containing 300 images.
### 2. Pre-Processing the Dataset
Rename the folders with names to numbers from 0 - 45 so as to label the dataset inorder to train and test the Neural Network.<br/>
The detailed information on renaming the dataset is given in [**info.txt**](https://github.com/SaiPrakashGit/PhotoOCR-Devanagari_Script_Recognition/blob/main/info.txt) file in this repository.
### 3. Converting (Unrolling) images to arrays and writing to csv files
Convert the images in both subfolders into arrays in dataframe with [**images_to_csv.py**](https://github.com/SaiPrakashGit/PhotoOCR-Devanagari_Script_Recognition/blob/main/images_to_csv.py) script by specifying the argument as below.<br/>
In Terminal : ***python images_to_csv.py TRAIN*** and ***python images_to_csv.py VALID*** for both folders.<br/>
Now, the csv files are ready to train and test the Neural Network against correct labels.
### 4. Training and Testing the Neural Network
The training has been done using 3 different Neural Networks in order to completely understand what is hindering the models' accuracy.<br/>
One Neural Network to classify just the digits (numbers).<br/>
Another Neural Network to classify just the characters (letters).<br/>
Another Neural Network to classify the total script (numbers + letters).<br/>
The results obtained are shown below.

## Results
### 1. Accuracy of the Model v/s Learning Rate
Learning Rate is a very important parameter for any Machine Learning model. If it is too high or too low, the model may fail to optimize the Cost Function to the Global Minimum resulting in poor accuracy.<br/>
Different learning rates are tried and their corresponding model's performance is plotted below.
<img src="https://raw.githubusercontent.com/SaiPrakashGit/PhotoOCR-Devanagari_Script_Recognition/main/plots/accuracy_vs_learningrate.png"><br/>
We can see that the model hits the sweetspot at learning rate = 0.05. Hence, the networks have been trained with this learning rate for 5 epochs.
### 2. Accuracy of the Digit Recognizing Model
Surprisingly, the model performed astonishingly well in recognizing the devanagari digits.<br/>
Accuracy after epoch 1 : 97.3 %<br/>
Accuracy after epoch 5 : 98.4 %<br/>
<img src="https://raw.githubusercontent.com/SaiPrakashGit/PhotoOCR-Devanagari_Script_Recognition/main/plots/digitaccuracy_vs_numberofepochs.png"><br/>
### 3. Accuracy of the Character Recognizing Model
Results for Character Recognition Model are shown below.<br/>
Accuracy after epoch 1 : 79.19 %<br/>
Accuracy after epoch 5 : 87.86 %<br/>
<img src="https://raw.githubusercontent.com/SaiPrakashGit/PhotoOCR-Devanagari_Script_Recognition/main/plots/characteraccuracy_vs_numberofepochs.png"><br/>
### 4. Accuracy of the Overall Script Recognizing Model
Results for Overall Script Recognition Model are shown below.<br/>
Accuracy after epoch 1 : 80.81 %<br/>
Accuracy after epoch 5 : 87.45 %<br/>
<img src="https://raw.githubusercontent.com/SaiPrakashGit/PhotoOCR-Devanagari_Script_Recognition/main/plots/overallaccuracy_vs_numberofepochs.png"><br/>
We can see that even if the digit recognition model performed astonishingly well in classifying digits, the overall multiclass classifier model's accuracy is hindered by the characters.

## Key Takeaways
After 5 epochs of training ,<br/>
* Devanagari Digit Recognition Model's Accuracy     : 98.40 %<br/>
* Devanagari Character Recognition Model's Accuracy : 87.86 %<br/>
* Devanagari Script Recognition Model's Accuracy    : 87.45 %<br/>
