import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# MLP 모델 생성
def create_mlp(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(1024, activation='relu'),    # 히든 레이어 1(1024)
        keras.layers.Dense(512, activation='relu'),     # 히든 레이어 2(512)
        keras.layers.Dense(256, activation='relu'),     # 히든 레이어 3(512),
        keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    # 모델 구조 확인
    model.summary()
    
    # 모델 컴파일
    model.compile(optimizer='adam',\
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# MLP 모델 학습
def train_mlp(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("테스트 정확도 (MLP):", test_acc)

    return history

# 손실 및 정확도 그래프
def plot_loss_and_accuracy(history):
    plt.figure(figsize=(14, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째 그래프
    plt.plot(history.history['loss'], 'b-', label='Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째 그래프
    plt.plot(history.history['accuracy'], 'r-', label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 그래프 출력
    plt.tight_layout()
    plt.show()
    
# 혼동 행렬 시각화
def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (MLP)')
    plt.show()