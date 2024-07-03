import keras
from keras import layers
from test2024_07_03_1049.Patches  import Patches  
from test2024_07_03_1049.PatchEncoder import PatchEncoder

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

print("\nBuild the ViT model")
class ViTModel():
    def __init__(self,data_augmentation):
        self.model=self.create_vit_classifier(data_augmentation)
    def create_vit_classifier(self,data_augmentation):
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
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

if __name__ == "__main__":#用于测试
    print()