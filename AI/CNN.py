import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# CNN 모델 생성
def create_cnn(input_shape, num_classes):
    model = keras.models.Sequential([
        # Conv + MaxPooling 1(32)
        keras.layers.Conv2D(input_shape = input_shape,
                            kernel_size = (3, 3), padding = 'same',
                            filters = 32),
        keras.layers.MaxPooling2D((2, 2)),

        # Conv + MaxPooling 2(64)
        keras.layers.Conv2D(input_shape = input_shape,
                            kernel_size = (3, 3), padding = 'same',
                            filters = 64),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Conv + MaxPooling 3(32)
        keras.layers.Conv2D(input_shape = input_shape,
                            kernel_size = (3, 3), padding = 'same',
                            filters = 32),
        # keras.layers.MaxPooling2D((2, 2)),

        # Flatten 및 완전 연결층
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(32, activation = 'relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # 모델 구조 확인
    model.summary()

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# CNN 모델 학습
def train_cnn(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    print("신경망 모델 학습 결과:")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print("테스트 정확도:", test_acc)

    return history