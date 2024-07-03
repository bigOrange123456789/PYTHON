import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt
print("\nImplement patch creation as a layer")
from test2024_07_03_1049.Patches  import Patches

print("\nBuild the ViT model")
print("Implement multilayer perceptron (MLP)")
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
from test2024_07_03_1049.ViTModel import ViTModel

def run_experiment(model,x_train,y_train,x_test,y_test):
    print("\n.Configure the hyperparameters")
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
    print("\nnum_epochs:",num_epochs)

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

class Main:
    def __init__(self):
        import sys
        print("Python运行环境:",sys.version)
        
        print("\nI.Prepare the data")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
        
        print("\nII.Configure the hyperparameters")
        image_size = 72  # We'll resize input images to this size
        patch_size = 6  # Size of the patches to be extract from the input images4

        print("\nIII.Use data augmentation")
        data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data for normalization.
        data_augmentation.layers[0].adapt(x_train)

        
        
        print("\nV.Let's display patches for a sample image")
        plt.figure(figsize=(4, 4))
        image = x_train[np.random.choice(range(x_train.shape[0]))]
        plt.imshow(image.astype("uint8"))
        plt.axis("off")

        resized_image = ops.image.resize(
            ops.convert_to_tensor([image]), size=(image_size, image_size)
        )
        patches = Patches(patch_size)(resized_image)
        print(f"Image size: {image_size} X {image_size}")
        print(f"Patch size: {patch_size} X {patch_size}")
        print(f"Patches per image: {patches.shape[1]}")
        print(f"Elements per patch: {patches.shape[-1]}")

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
            plt.axis("off")

        print("\nVI.Compile, train, and evaluate the model")
        vit=ViTModel(data_augmentation)
        vit_classifier = vit.model#create_vit_classifier(data_augmentation)
        history = run_experiment(vit_classifier,x_train,y_train,x_test,y_test)

        def plot_history(item):
            plt.plot(history.history[item], label=item)
            plt.plot(history.history["val_" + item], label="val_" + item)
            plt.xlabel("Epochs")
            plt.ylabel(item)
            plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
            plt.legend()
            plt.grid()
            plt.show()
        plot_history("loss")
        plot_history("top-5-accuracy")
        
if __name__ == "__main__":#用于测试
    Main()