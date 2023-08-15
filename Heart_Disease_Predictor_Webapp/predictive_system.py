# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model.
loaded_model = pickle.load(open('/Users/ellietripathi/Downloads/trained_model.sav', 'rb'))

input_data = (60,1,0,117,230,1,1,160,1,1.4,2,2,3,) #(56,0,1,140,294,0,0,153,0,1.3,1,0,2)

# change the input data to numpy array.
input_data = np.asarray(input_data)

# making prediction for this particular input.
input_data_reshaped = input_data.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("You dont have a Heart Disease.")
else:
    print("You have a Heart Disease.")