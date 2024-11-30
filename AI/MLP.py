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
        keras.layers.Dense(512, activation='relu'),     # 히든 레이어 3(512)
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

# 혼동 행렬 시각화
def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (MLP)')
    plt.show()