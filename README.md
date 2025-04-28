# MNIST Handwritten Digit Recognition

This project implements a neural network to classify handwritten digits (0–9) from the MNIST dataset using TensorFlow and Keras.

## Tech Stack

- Python 3.x  
- Pandas, NumPy  
- TensorFlow / Keras  
- Scikit-learn
- MatplotLib, Seaborn 

## Dataset

- MNIST Handwritten Digits Dataset
- 28x28 grayscale images (flattened to 784 features)
- 60,000 training samples, 10,000 test samples
- Labels are digits from 0 to 9
- Source: [Kaggle - MNIST CSV Format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## Workflow
1. **Importing Libraries and Loading Data**  
   Loading and inspecting the MNIST dataset from CSV to understand its structure.

2. **Data Visualization**  
Using the `show_digit(index)` to display sample handwritten digits with Matplotlib.
```python
def show_digit(index):
    pixels = 255 - df.iloc[index, 1:].values 
    image = pixels.reshape(28, 28).T
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {df.iloc[index, 0]}")
    plt.axis('off')
    plt.show()
```

3. **Data Preprocessing**  
   - Normalizing pixel values to [0,1] using one-hot encode.
 ```python
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
```
   - Split data into training and testing sets.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=123)
```

4. **Building the Neural Network**
   Creating a feedforward neural network with:
   - Input layer: 784 neurons (flattened 28x28 image)
   - Hidden layer: 30 neurons with ReLU activation
   - Output layer: 10 neurons with Softmax activation(Digits 0-9)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

5. **Training the Model**  
   Train the model using categorical cross-entropy loss and Adam optimizer.
```python
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
```
6. **Evaluating the Model**  
   Assessing accuracy on both training and testing datasets.
```
Training accuracy: 97.78%
Test accuracy: 95.74%
```
7. **Making Predictions and Visualization**  
   Predicting labels on new samples and plot a confusion matrix to visualize performance.
   ```python
   def inspect(index):
    image = X[index].reshape(1, -1)
    prediction = model.predict(image, verbose=0)
    predicted_label = np.argmax(prediction)
    actual_label = np.argmax(y_encoded[index])
    print(f"Actual: {actual_label}, Predicted: {predicted_label}")
    show_digit(index)
   ```
### Confusion Matrix:
- ![image](https://github.com/user-attachments/assets/f06b8e76-49df-42b2-9b23-d76012972602)

