# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # Load dataset
# df = pd.read_csv('./titanic (1).csv')
# print(df.head())
# print(df.isnull().sum())

# # Check what columns are available
# print("All Columns in Dataset:")
# print(df.columns.tolist())  # See exact names

# # List of target columns to remove
# columns_to_drop = ['adult_male', 'embark_town', 'alive', 'who', 'alone']

# # Only drop columns that actually exist
# existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

# # Drop them
# df.drop(columns=existing_columns_to_drop, inplace=True)

# # Handle missing values
# df.dropna(inplace=True)
# print("\nAfter dropping rows with NaN values:")
# print(df.isnull().sum())

# # Label Encoding for categorical features
# label_encoder = LabelEncoder()
# categorical_columns = ['sex', 'embarked', 'deck', 'class']

# for col in categorical_columns:
#     df[col] = label_encoder.fit_transform(df[col])

# # Define the target variable (survived) and features
# X = df.drop('survived', axis=1)  # All columns except 'survived' are features
# y = df['survived']  # The 'survived' column is the target

# # Split the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=64)

# # Scale the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Create a more complex MLP model using Keras
# mlp_model = Sequential()

# # Add input layer (using the number of features as input size)
# mlp_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # More neurons in the input layer

# # Add hidden layers with Dropout for regularization
# mlp_model.add(Dense(64, activation='relu'))
# mlp_model.add(Dropout(0.3))  # Dropout to prevent overfitting

# mlp_model.add(Dense(64, activation='relu'))
# mlp_model.add(Dropout(0.3))

# # Add output layer (binary classification)
# mlp_model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# mlp_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# # Early stopping to avoid overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# mlp_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# # Evaluate the model on the test set
# loss, accuracy = mlp_model.evaluate(X_test, y_test)

# print(f"MLP Model Accuracy: {accuracy}")

# # If accuracy is within the desired range (70-80%), save the model
# if 0.70 <= accuracy <= 0.80:
#     mlp_model.save('titanic_mlp_model.h5')
#     print("Model saved to 'titanic_mlp_model.h5' because accuracy is within the desired range.")
# else:
#     print(f"Model did not achieve the desired accuracy (70-80%). Current accuracy: {accuracy:.4f}")
