import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from MLP import create_mlp, train_mlp

# CNN 모델 생성
def create_cnn(input_shape):
    model = keras.models.Sequential([
        # Conv + MaxPooling 1(32 filters)
        keras.layers.Conv2D(input_shape = input_shape, kernel_size = (3, 3), padding = 'same', filters = 32),
        keras.layers.MaxPooling2D((2, 2)),

        # Conv + MaxPooling 2(64 filters)
        keras.layers.Conv2D(input_shape = input_shape, kernel_size = (3, 3), padding = 'same', filters = 64),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Conv 3(32 filters)
        keras.layers.Conv2D(input_shape = input_shape, kernel_size = (3, 3), padding = 'same', filters = 32),
    ])
    
    # 모델 구조 확인
    model.summary()
    
    return model

# 머신러닝 모델 학습 및 평가
def train_cnn(model, X_train, y_train, X_test, y_test):
    # Flatten 레이어 출력 계산
    flatten_model = keras.Sequential([model, keras.layers.Flatten()])
    X_train_flatten = flatten_model.predict(X_train)
    X_test_flatten = flatten_model.predict(X_test)
    
    # 원핫 인코딩 -> 정수 레이블 변환
    y_train_labels = y_train.argmax(axis=1)
    y_test_labels = y_test.argmax(axis=1)

    results = {}

    # KNN
    print("\nKNN 학습 중...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_flatten, y_train_labels)
    knn_pred = knn.predict(X_test_flatten)
    results['KNN'] = accuracy_score(y_test_labels, knn_pred)
    plot_confusion_matrix(y_test_labels, knn_pred, "KNN")
    
    # SVM
    print("\nSVM 학습 중...")
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train_flatten, y_train_labels, test_size=0.9, random_state=42)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train_sample, y_train_sample)
    svm_pred = svm.predict(X_test_flatten)
    results['SVM'] = accuracy_score(y_test_labels, svm_pred)
    plot_confusion_matrix(y_test_labels, svm_pred, "SVM")

    # Decision Tree
    print("\nDecision Tree 학습 중...")
    dt = DecisionTreeClassifier()
    dt.fit(X_train_flatten, y_train_labels)
    dt_pred = dt.predict(X_test_flatten)
    results['Decision Tree'] = accuracy_score(y_test_labels, dt_pred)
    plot_confusion_matrix(y_test_labels, dt_pred, "Decision Tree")
    
    # MLP
    print("\nMLP 학습 중...")
    mlp = create_mlp(X_train_flatten.shape[1:], num_classes=y_train.shape[1])
    train_mlp(mlp, X_train_flatten, y_train, X_test_flatten, y_test, epochs=10, batch_size=64)
    mlp_pred = mlp.predict(X_test_flatten)
    mlp_pred_labels = mlp_pred.argmax(axis=1)
    results['MLP'] = accuracy_score(y_test_labels, mlp_pred_labels)
    plot_confusion_matrix(y_test_labels, mlp_pred_labels, "MLP")

    # 결과 출력
    for model_name, accuracy in results.items():
        print(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")

    return results


# 성능 비교 시각화
def plot_model_comparison(results):
    model_names = list(results.keys())
    accuracies = [results[name] * 100 for name in model_names]

    plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Accuracies')
    plt.ylim(0, 100)
    plt.show()
    

# 혼동 행렬 시각화
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.show()