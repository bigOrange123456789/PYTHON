import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt
from Patches  import Patches
from ViTModel import ViTModel

class Main:
    def __init__(self):
        import sys
        print("Python运行环境:",sys.version)
        
        print("\nI.Prepare the data")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
        print(f"x_test shape : {x_test.shape}  - y_test  shape: {y_test.shape} ")

        print("\nII.Use data augmentation")
        image_size = 72  # We'll resize input images to this size
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

        self.displayPatches(x_train,image_size)

        print("\nIV.Compile, train, and evaluate the model")
        vit=ViTModel(data_augmentation)
        exit(0)
        vit.train(x_train,y_train)
        vit.predict(x_test,y_test)#evaluate
    
    def displayPatches(self,x_train,image_size):
        print("\nIII.Let's display patches for a sample image")
        plt.figure(figsize=(4, 4))
        image = x_train[np.random.choice(range(x_train.shape[0]))]#image: (32, 32, 3)<class 'numpy.ndarray'>
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
        resized_image = ops.image.resize(
            ops.convert_to_tensor([image]), size=(image_size, image_size)
        )
        patch_size = 6  # Size of the patches to be extract from the input images4
        print(f"resized_image: {resized_image.shape}{type(resized_image)}")#(1, 72, 72, 3) <class 'tensorflow.python.framework.ops.EagerTensor'>
        patches = Patches(patch_size)(resized_image)
        print(f"Image size: {image_size} X {image_size}") # Image size: 72 X 72
        print(f"Patch size: {patch_size} X {patch_size}") # Patch size: 6 X 6
        print(f"Patches per image: {patches.shape[1]}")   # Patches per image: 144
        print(f"Elements per patch: {patches.shape[-1]}") # Elements per patch: 108

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
            plt.axis("off")

        
if __name__ == "__main__":#用于测试
    Main()