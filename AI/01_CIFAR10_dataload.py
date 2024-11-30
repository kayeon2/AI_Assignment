import tensorflow as tf
import matplotlib.pyplot as plt

from MLP import create_mlp, train_mlp

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])  # CIFAR-10 클래스 이름 표시
    plt.axis('off')
plt.show()

# 정규화
X_train, X_test = X_train / 255, X_test / 255  

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

input_shape = X_train.shape[1:]
num_classes = 10
mlp_model = create_mlp(input_shape, num_classes)
history = train_mlp(mlp_model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64)
