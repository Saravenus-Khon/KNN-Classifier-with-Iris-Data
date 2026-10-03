"""



"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




# load the iris dataset and give each column as name for: SepalLength,SepalWidth,PetalLength,PetalWidth,Species
iris = pd.read_csv("iris.csv")

print(iris.head())


X = iris[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]    # these are the input features
y = iris['Name']       # this is the target


# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# create the KNN classifier with 5 nearest neighbors
model = KNeighborsClassifier(n_neighbors=5)


# train the model
model.fit(X_train, y_train)


# Make predictions using the test data
y_prediction = model.predict(X_test)

# Calculate how accurate the model predicted the test flowers
accuracy = accuracy_score(y_test, y_prediction)

print("\nTest Predictions:")
print(y_prediction)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Ask the user for the 4 flower measurements
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))


# Put these four values into the same feature order as the training data using a data frame
new_flower = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns= ['SepalLength','SepalWidth','PetalLength','PetalWidth'])


# Predict the Iris Species
prediction = model.predict(new_flower)
print("The predicted Iris Type: ", prediction[0])

# Make a scatter plot showing iris species by petal measurements


# Make a scatter plot showing iris species by petal measurements

species_colors = {
    'Iris-setosa': 'orange',
    'Iris-versicolor': 'purple',
    'Iris-virginica': 'green'
}

# Plot each iris species
for species, color in species_colors.items():
    flower_grp = iris[iris['Name'] == species]

    plt.scatter(
        flower_grp['PetalLength'],
        flower_grp['PetalWidth'],
        color=color,
        label=species
    )

# Plot the user's new flower
plt.scatter(
    petal_length,
    petal_width,
    color='black',
    marker='X',
    s=150,
    label='New Flower'
)

# Label the graph
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("KNN Iris Classification")
plt.legend()

# Show the graph
plt.show()