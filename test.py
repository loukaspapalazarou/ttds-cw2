import pickle


def save_function(func, filename):
    with open(filename, "wb") as file:
        pickle.dump(func, file)


def load_function(filename):
    with open(filename, "rb") as file:
        func = pickle.load(file)
    return func


# Example function to save and load
def example_function(x):
    return x * 2


# Save the function to a file
save_function(example_function, "example_function.pkl")

# Load the function from the file
loaded_function = load_function("example_function.pkl")

# Test the loaded function
result = loaded_function(5)
print(result)
