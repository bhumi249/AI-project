#step1=> make datasets

import pandas as pd
import numpy as np

# Number of samples you want to generate
num_samples = 1000

# Generate data for each column
data = {
    "sample_id": range(1, num_samples + 1),  # Sample ID
    "pH": np.random.uniform(6.0, 8.5, num_samples),  # pH level of water
    "turbidity": np.random.uniform(1.0, 5.0, num_samples),  # Turbidity (clarity)
    "dissolved_oxygen": np.random.uniform(5.0, 9.0, num_samples),  # Oxygen level in water
    "temperature": np.random.uniform(10.0, 35.0, num_samples),  # Temperature of water in Celsius
    "purity_level": np.random.choice(["Pure", "Impure"], num_samples)  # Water purity level
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("water_dataset.csv", index=False)

# Display the first few rows of the dataset
print(df.head())


#step=>2 prepare dataset for testing
# Load the dataset
df = pd.read_csv('water_dataset.csv')

# Convert categorical labels (Pure/Impure) to numeric values (0 and 1)
df['purity_level'] = df['purity_level'].map({'Pure': 0, 'Impure': 1})

# Features (independent variables)
X = df[['pH', 'turbidity', 'dissolved_oxygen', 'temperature']]

# Target variable (dependent variable)
y = df['purity_level']

# Split the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#step3=>
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#step=>4
from sklearn.metrics import accuracy_score

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

#step=>5 Make predictions on new data
# Example of a new sample
new_sample = pd.DataFrame({
    'pH': [7.2],
    'turbidity': [2.5],
    'dissolved_oxygen': [7.8],
    'temperature': [25.0]
})

# Predict the purity level
prediction = model.predict(new_sample)

# Output the result
print("Predicted purity level:", "Pure" if prediction == 0 else "Impure")

#step6=>Save the model
import joblib

joblib.dump(model, 'water_quality_model.pkl')

# Later, you can load the model like this
# model = joblib.load('water_quality_model.pkl')

# Step 7 => Plot Accuracy Graph Over Multiple Runs
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Initialize an empty list to store accuracy scores for each run
accuracy_scores = []

# Run the training and testing process multiple times (e.g., 10 times) to gather accuracy data
for i in range(10):  # 10 runs
    # Re-train the model each time
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)  # Append each accuracy score to the list

    print(f'Run {i + 1} - Accuracy: {accuracy * 100:.2f}%')

# Plot the accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), accuracy_scores, marker='o', color='b', linestyle='-')
plt.title('Model Accuracy Over Multiple Runs')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.grid(True)
plt.show()


