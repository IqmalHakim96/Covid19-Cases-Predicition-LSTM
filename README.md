# Covid19-Cases-Predicition-LSTM

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

A model prediction of Covid19 new cases in Malaysia using deep learning with approach Long Short Term Memory (LSTM) Neural Network

### Introduction 
OBJECTIVE - To create a deep learning model to predict the new_cases of Covid-19 in Malaysia by using LSTM neural network and using past 30 days of number of cases.


In this analysis, there are 2 dataset were used to predict number of new_cases.
   1. cases_malaysia_train.csv - The data are from date 25/1/2020 to 4/12/2021 with 680 data entry and 31 columns
   2. cases_malaysia_test.csv - The data are from date 5/12/2021 to 14/3/2022 with 100 data entry and 31 columns\

In this case, we only want to predict new_cases, hence we only focused on "cases_new" column only. Regarding on inspection of the data, there are some missing values/'Nan' and symbol found in the data. To make the data looking neat and easy to do next step of making model, the data need to clean first in cleaning process.


### Deep Learning Model Using LSTM Layer

![Model](https://user-images.githubusercontent.com/105650253/211257569-030fd583-9809-4932-bb8c-f7254299a451.PNG)

![Model_Architecture](https://user-images.githubusercontent.com/105650253/211257898-3b98d2c8-d97f-4b78-b942-d9893c4ed5d8.PNG)

Using 2 Dropout layers may reducing the overfitting of the dataset. The LSTM nodes was used in 64 nodes with the depht of 2. The epochs was used is 100 number of epochs to train the data:

![epoch both](https://user-images.githubusercontent.com/105650253/211258429-733c32f7-93bd-448c-80b1-30bc2f2bef9e.PNG)

### Result of prediction vs actual
![output](https://user-images.githubusercontent.com/105650253/211258534-4309d6e8-7e98-4cc0-b869-7783c9d2e71d.png)

By using the created model, The Mean Absolute Percentage Error is in 0.2%. To conclude this, this model are good enough to be used for future prediction.

### Discussion

1. The created model is good enough to use for next predicition eventhough with lot of data
2. The overfitting can be fix by increasing the number of epochs, parameters of Dropout layers, so that the data can be trained well
3. The LSTM model can be better by increasing number of nodes, add more layers such as Batch Normalization() 

### Acknowledgement
The dataset used in this analysis is from : https://github.com/MoH-Malaysia/covid19-public
