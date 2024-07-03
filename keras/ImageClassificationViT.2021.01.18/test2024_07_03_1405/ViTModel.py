import keras
from keras import layers
from Patches  import Patches  
from PatchEncoder import PatchEncoder

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

print("\nBuild the ViT model")
class ViTModel():
    def __init__(self,data_augmentation):
        self.model=self.__initModel(data_augmentation)
        self.__compile()
        self.checkpoint_filepath = "./checkpoint.weights.h5"#"/tmp/checkpoint.weights.h5"
        
    def __initModel(self,data_augmentation):
        num_classes = 100
        input_shape = (32, 32, 3)
        
        print("\n.Configure the hyperparameters")
        image_size = 72  # We'll resize input images to this size
        patch_size = 6  # Size of the patches to be extract from the input images
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        transformer_layers = 8
        mlp_head_units = [
            2048,
            1024,
        ]  # Size of the dense layers of the final classifier

        inputs = keras.Input(shape=input_shape)
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        return keras.Model(inputs=inputs, outputs=logits)

    def __compile(self):
        learning_rate = 0.001
        weight_decay = 0.0001

        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

    def train(self,x_train,y_train):
        batch_size = 256
        num_epochs = 1# 10  # For real training, use num_epochs=100. 10 is a test value
        print("\nnum_epochs:",num_epochs)
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            self.checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,#随机划出验证集
            callbacks=[checkpoint_callback],
        )
        self.analysis(history)
    
    def analysis(self,history):
        def plot_history(item):
            import matplotlib.pyplot as plt
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

    def predict(self,x_test,y_test):
        self.model.load_weights(self.checkpoint_filepath)
        _, accuracy, top_5_accuracy = self.model.evaluate(x_test, y_test)#最终的测试集
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

if __name__ == "__main__":#用于测试
    print()