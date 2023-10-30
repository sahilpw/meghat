# dataset https://www.kaggle.com/competitions/titanic/overview

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the Titanic dataset (replace 'titanic.csv' with the actual file path)
file_path = '/content/train.csv'
titanic_data = pd.read_csv(file_path)
titanic_data.dropna(inplace=True)
# Data Preprocessing
# Handle missing values, encoding categorical variables, etc.
titanic_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
titanic_data.dropna(subset=['Embarked'], inplace=True)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])

# Select features and target variable
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)


# Plot the count of survivors
plt.figure(figsize=(6, 4))
survived_counts = y_test.value_counts()
survived_counts.plot(kind='bar', color=['blue', 'red'])
plt.title('Survivors (0 = Did Not Survive, 1 = Survived)')
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
