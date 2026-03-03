                
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

print("Loading data from E:\\MNIST...")
data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape

np.random.shuffle(data) 

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
_, m_train = X_train.shape

# math ftns
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            print(f"Accuracy: {get_accuracy(predictions, Y):.4f}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, current_image)
    prediction = get_predictions(A2)
    label = Y_train[index]
    
    print(f"Prediction: {prediction}, Label: {label}")
    
    plt.gray()
    plt.imshow(current_image.reshape((28, 28)), interpolation='nearest')
    plt.show()

for i in range(3):
    test_prediction(i, W1, b1, W2, b2)

def interactive_demo(W1, b1, W2, b2):
    print("\n" + "="*40)
    print("      MNIST NEURAL NETWORK LAB")
    print("="*40)
    print("Commands: [Enter] for Random Test | [q] to Quit")

    while True:
        user_input = input("\n> ")
        if user_input.lower() == 'q':
            print("Exiting Lab. Great session!")
            break
        
        idx = np.random.randint(0, m_train)
        
      
        current_img = X_train[:, idx, None]
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, current_img)
        prediction = get_predictions(A2)[0]
        actual = Y_train[idx]

        status = " CORRECT" if prediction == actual else "INCORRECT"
        
        print(f"Index: {idx}")
        print(f"Network Prediction: {prediction}")
        print(f"Actual Label:       {actual}")
        print(f"Result:             {status}")

        plt.figure(figsize=(4,4))
        plt.imshow(current_img.reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {prediction} | Actual: {actual}")
        plt.axis('off')
        plt.show()

interactive_demo(W1, b1, W2, b2)
