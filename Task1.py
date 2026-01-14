# Iris Flower Classification using Given Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\Admin\Desktop\Iris.csv")

# Drop Id column if present
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Encode Species column
df['Species'] = df['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

# Features and target
X = df.drop('Species', axis=1)
y = df['Species']


# Class Distribution
plt.figure()
df['Species'].value_counts().plot(kind='bar')
plt.title("Class Distribution of Iris Species")
plt.xlabel("Species (0:Setosa, 1:Versicolor, 2:Virginica)")
plt.ylabel("Count")
plt.show()

# Sepal Length Distribution
plt.figure()
plt.hist(df['SepalLengthCm'])
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Petal Length Distribution
plt.figure()
plt.hist(df['PetalLengthCm'])
plt.title("Petal Length Distribution")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

k_values = range(1, 11)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))


plt.figure()
plt.plot(k_values, accuracies)
plt.title("Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.show()


best_k = k_values[accuracies.index(max(accuracies))]
print("Best K Value:", best_k)
print("Highest Accuracy:", max(accuracies))

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
