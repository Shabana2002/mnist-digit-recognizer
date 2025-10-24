from tensorflow.keras.models import load_model

# Load the model
model = load_model('model/mnist_cnn.h5')
print("Model loaded successfully!")

# Optional: check a summary of the model
model.summary()
