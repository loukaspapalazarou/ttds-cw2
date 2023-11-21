from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

# Load a sample dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create an SVM model
svm = SVC(kernel="linear")

# Set the number of training iterations (epochs)
n_epochs = 10000

# Create a tqdm progress bar
with tqdm(total=n_epochs, desc="Training SVM") as pbar:
    for epoch in range(n_epochs):
        # Fit the SVM model for one epoch
        svm.fit(X_train, y_train)

        # Update the progress bar
        pbar.update(1)

# After training is complete, you can use the trained model for predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.2f}")
