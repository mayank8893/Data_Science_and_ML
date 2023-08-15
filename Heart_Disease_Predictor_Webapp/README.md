# Heart_Disease_Predictor_Webapp

In this project, I am looking at **the heart disease dataset from kaggle**. This dataset has information on a person age, sex, cholestrol and other heart health metrics from EKG etc. The dataset also has a target variable which which tells if the person has heart issues (1) or if they have a healthy heart (0). **The aim of this project is to build a classification model that tells whether a person has healthy heart or not**.

I will first load the dataset, get some statistical info on it and clean it. I will then split it into a training and a test set. Then I will train a **Logistic Regression model on the test set and use it to make prediction about the test set**. I have also trained a bunch of other model, but there was no noticeable improvement in accuracy. The model works with an **accuracy of 80% on the test set**. Finally, I have made a **predictive system** where you can input the metrics for your heart and the model will predict if you have a good heart.

I have then saved this model and deployed it to webapp using **streamlit**.

<img width="550" alt="Screen Shot 2023-08-15 at 10 45 31 AM" src="https://github.com/mayank8893/Heart_Disease_Predictor_Webapp/assets/69361645/1d4d4fac-7722-4a75-a8f2-6d5886fc0fbb">
<img width="947" alt="Screen Shot 2023-08-15 at 10 45 41 AM" src="https://github.com/mayank8893/Heart_Disease_Predictor_Webapp/assets/69361645/a8349a06-b9ac-481b-b5b5-74e34de326b0">


![Screen Shot 2023-08-15 at 10 47 29 AM](https://github.com/mayank8893/Heart_Disease_Predictor_Webapp/assets/69361645/f763fc8b-ee06-4518-8ff5-2574726423e0)
