import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load model
model = load_model('model/mnist_cnn.h5')

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Pick a sample
index = 0  # change 0-9999 to try different images
sample_image = x_test[index:index+1]

# Make prediction
predicted_class = np.argmax(model.predict(sample_image, verbose=0), axis=1)

# Show result
print(f"Predicted: {predicted_class[0]}")
print(f"Actual: {y_test[index]}")
