import tensorflow as tf
import matplotlib.pyplot as plt
from MLP import create_mlp, train_mlp, plot_confusion_matrix, plot_loss_and_accuracy
from CNN import create_cnn, train_cnn, plot_model_comparison
import numpy as np

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

# 데이터 정규화
X_train, X_test = X_train / 255, X_test / 255  

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

input_shape = X_train.shape[1:]     # 입력 데이터의 형태
num_classes = 10                    # CIFAR-10 클래스 개수

# MLP
print("----- MLP 구현 -----")
mlp_model = create_mlp(input_shape, num_classes)
history = train_mlp(mlp_model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

# MLP 예측 결과 생성
y_pred = mlp_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# 혼동 행렬 시각화
plot_confusion_matrix(y_test_labels, y_pred_labels)

# 손실 그래프
plot_loss_and_accuracy(history)

# CNN
print("----- CNN 구현 -----")
cnn_model = create_cnn(input_shape)
results = train_cnn(cnn_model, X_train, y_train, X_test, y_test)
plot_model_comparison(results)
