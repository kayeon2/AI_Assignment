import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from MLP import create_mlp, train_mlp, plot_confusion_matrix, plot_loss_and_accuracy
from CNN import create_cnn, train_cnn, plot_model_comparison

IMAGE_SIZE = (128,128)

def load_train_data(folder_path):
    X = []
    y = []
    class_names = os.listdir(folder_path)
    print(class_names)

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names


# Load image data
def load_test_data(folder_path):
    X = []
    filenames = []
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            filenames.append(image_name)
    X = np.array(X)
    return X, filenames

# # Load training and testing data
train_folder = './flowers-dataset/train'
test_folder = './flowers-dataset/test'
X_train, y_train, class_names = load_train_data(train_folder)
X_test, test_filenames = load_test_data(test_folder)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize values
X_train = X_train / 255.0
X_test = X_test / 255.0

print("X_train_split.shape:", X_train.shape)
print("y_train_split.shape:", y_train.shape)
print("X_test_split.shape:", X_test.shape)
print("y_test_split.shape:", y_test.shape)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i]])  # 이미지에 해당하는 클래스 이름 표시
    plt.axis('off')
plt.show()

# One-hot Encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

input_shape = X_train.shape[1:]
num_classes = len(class_names)

# MLP 모델
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

# CNN 모델
print("----- CNN 구현 -----")
cnn_model = create_cnn(input_shape)

# 머신러닝 모델 성능 비교
results = train_cnn(cnn_model, X_train, y_train, X_test, y_test)
plot_model_comparison(results)