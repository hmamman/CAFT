import argparse
import sys
import os

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Get the absolute path to the directory where main.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../"))

from utils import helpers


def create_dnn_model(input_shape=(None, 13), nb_classes=2):
    model = Sequential([
        Input(shape=input_shape[1:]),
        Dense(30, activation="relu"),
        Dense(20, activation="relu"),
        Dense(15, activation="relu"),
        Dense(10, activation="relu"),
        Dense(5, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def train_and_evaluate(data, dataset_name):
    X, y, input_shape, nb_classes = data()

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

    model = create_dnn_model(input_shape=input_shape)

    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[es],verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print(f"DNN Model's accuracy on {dataset_name} dataset is: {accuracy:.4f}")

    # Save the model to disk
    dir_path = f"models/{dataset_name}"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f'{dir_path}/dnn.keras'
    model.save(model_path)
    print(f"DNN Model saved to {model_path}")
    #           Bank    Credit  Census  Compas  Meps
    #   DNN:    0.8973  0.7417  0.8465  0.8228  0.8775


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on a given dataset.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.")

    args = parser.parse_args()

    configs = helpers.get_config_dict()

    dataset_name = args.dataset_name.lower()

    data_path = f"data/{dataset_name}.py"
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    config = configs[dataset_name]

    data = helpers.get_data(config.dataset_name)

    train_and_evaluate(data, dataset_name)


if __name__ == "__main__":
    main()
