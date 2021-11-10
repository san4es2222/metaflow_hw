from metaflow import FlowSpec, step, conda_base

@conda_base(libraries={'tensorflow':'2.7.0', 'numpy':'1.19.5', 'matplotlib': '3.2.2'})
class Autoencoder(FlowSpec):
    @step
    def start(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras import layers

        input = layers.Input(shape=(28, 28, 1))

        # Encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decoder
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

        # Autoencoder
        self.autoencoder = Model(input, x)
        self.autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
        self.autoencoder.summary()

        self.next(self.define_data)

    @step
    def define_data(self):
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from tensorflow.keras.datasets import mnist

        from utils import preprocess, noise, display

        (train_data, _), (test_data, _) = mnist.load_data()

        self.train_data = preprocess(train_data)
        self.test_data = preprocess(test_data)

        # Create a copy of the data with added noise
        self.noisy_train_data = noise(self.train_data)
        self.noisy_test_data = noise(self.test_data)

        # Display the train data and a version of it with added noise
        display(self.train_data, self.noisy_train_data)

        self.datasets = [[self.train_data, self.test_data],
                          [self.noisy_train_data, self.noisy_test_data]]

        self.next(self.fit_predict, foreach='datasets')

    @step
    def fit_predict(self):
        self.data_train, self.data_test = self.input

        self.autoencoder.fit(
            x=self.data_train,
            y=self.data_train,
            epochs=50,
            batch_size=128,
            shuffle=True,
            validation_data=(self.data_test, self.data_test),
        )
        
        predictions = self.autoencoder.predict(self.data_test)
        display(self.data_test, predictions)

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    Autoencoder()   
