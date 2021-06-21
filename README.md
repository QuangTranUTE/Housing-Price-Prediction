# Housing Price Prediction

## What is this?
This repository consists of end-to-end code for processing and training regressors. It also includes trained models for predicting housing price in Ho Chi Minh city, Vietnam. Please see below sections for information about data and trained models.

## How to use it
#### *1. Using it as a housing price estimator* 
Simply go to this [**Google Colab**](https://colab.research.google.com/drive/1w-ISNEscUq40rGm2gld8_fKZSLBkqMf4?usp=sharing), then click **Run cell** (Ctrl-Enter), input apartment info and press **ESTIMATE PRICE**.

![Demo using the trained model on Colab](/resources/demo.PNG "Hope you enjoy it!") 
**Disclaimer:** Currently, this repository serves research and educational purpose only. No guarantee of accuracy for other purposes.

#### *2. Using `end_to_end_regression.py` to create your own models*
Instructions are included in the file.

At the moment, the code includes 4 models, namely `Linear regression, Polinomial regression, Decision tree regression, Random forest regression`. However, you can easily add models, such as `Support vector regression, Neural networks`, by adding them in PART 5 of the code. Please find more detailed instructions in the code file. 

After training, you can deploy your model on, for example, Google Colab as the example colab given above.  

## Brief info about the training data and trained model 
The model deployed in this [**Google Colab**](https://colab.research.google.com/drive/1w-ISNEscUq40rGm2gld8_fKZSLBkqMf4?usp=sharing) is a Random forest regressor (file `SOLUTION_model.pkl` in the `models` folder). This model and other models given in the `models` folder are trained and fine-tuned using a dataset consisting of about 2000 samples of apartment info and price in Hochiminh city. 

The training data are taken from laydulieu.com in June 2021 (see `datasets` folder for details). 

## Reference
Some parts of `end_to_end_regression.py` are based on the code given in Chapter 2 of this very well-written book about machine learning: [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron. I would highly recommend it to anyone who wish to learn about machine learning and deep learning.
