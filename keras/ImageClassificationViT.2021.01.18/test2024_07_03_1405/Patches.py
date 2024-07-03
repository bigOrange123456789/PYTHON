import keras
from keras import layers
from keras import ops

print("\nImplement patch creation as a layer")
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size 
        # print("self.patch_size",self.patch_size)
        # self.patch_size 6 

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        # print(f"[call]images: {images.shape} {type(images)}") 
        # images: (1, 72, 72, 3) <class 'tensorflow.python.framework.ops.EagerTensor'>
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        # print(f"[call]patches: {patches.shape} {type(patches)}") 
        # patches: (1, 12, 12, 108) <class 'tensorflow.python.framework.ops.EagerTensor'>
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        # print(f"[call]patches: {patches.shape} {type(patches)}") 
        # patches: (1, 144, 108) <class 'tensorflow.python.framework.ops.EagerTensor'>
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
