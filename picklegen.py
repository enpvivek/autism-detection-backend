import pandas as pd
import numpy as np

data = pd.read_csv("autism_screening.csv")

data['age'] = data['age'].fillna(round(data['age'].mean()))
data['ethnicity'] = data['ethnicity'].replace('?', 'others')

from sklearn.preprocessing import LabelEncoder

# Select the categorical columns for label encoding
categorical_cols = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'relation', 'Class/ASD']

# Perform label encoding
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

data = data.drop('age_desc', axis=1)

X = data.drop('Class/ASD', axis=1)
y = data['Class/ASD']   

# Splitting the dataset into train and test sets: 70-30 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, random_state = 12)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the input data for RNN
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the model architecture
model_rnn = Sequential([
    SimpleRNN(64, input_shape=(X_train_rnn.shape[1], 1), activation='relu'),
    Dense(1, activation='sigmoid')
])

model_rnn.summary()

# Compile the model
model_rnn.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model
history_rnn = model_rnn.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_split=0.2)


# Evaluate the model
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_rnn, y_test)
print(f'RNN: Loss: {loss_rnn}, Accuracy: {accuracy_rnn}')

import pickle

# As the performance model_rnn is the best model
best_model = model_rnn

# Save the model to a file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)