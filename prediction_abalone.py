import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

abalone_train = pd.read_csv(
  "abalone_data.csv",
  names=[
    "Length", "Diameter", "Height", "Whole weight", 
    "Shucked weight", "Viscera weight", "Shell weight", "Age"
  ]
)

print(abalone_train)

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')
print(abalone_features)
print(abalone_labels)

# convert to np array
abalone_features = np.array(abalone_features)
#abalone_labels = np.array(abalone_labels)

# implement model (using tf.keras) to predict age
model = tf.keras.Sequential([
  layers.Dense(units=64, activation="relu", input_shape=(abalone_features.shape[1],)),
  layers.Dense(units=1)
])

# compile the model here
model.compile(loss="mean_squared_error",optimizer="adam")

# fit the model here
ff = model.fit(abalone_features,abalone_labels,epochs=10, verbose=0)
print("ff::",ff)
print(model.summary())
print(ff.history)


