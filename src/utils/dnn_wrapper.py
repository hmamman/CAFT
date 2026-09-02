from sklearn.base import BaseEstimator, ClassifierMixin

import tensorflow as tf

class Sequential(BaseEstimator, ClassifierMixin):
    def __str__(self):
        return 'Sequential'

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        inp = tf.convert_to_tensor(X, dtype=tf.float32)
        prob = self.model(inp, training=False)  # Model output

        # Check if the output is a single-column (binary classification)
        if prob.shape[1] == 1:
            # Convert single-column probabilities to two columns [P(class 0), P(class 1)]
            prob = tf.concat([1 - prob, prob], axis=1)
        return prob.numpy()

    def predict(self, X):
        prob = self.predict_proba(X)
        return prob.argmax(axis=1)


def dnn_model_wrapper(model):
    if hasattr(model, 'layers'):
        return Sequential(model=model)
    return model