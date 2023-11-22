import numpy as np
import tensorflow as tf
import os
from random import sample, seed
from Utils import load_image, load_mask, calculate_iou, save_samples
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Read and prepare data.
images_path = ".\\data\\data\\images"
masks_path = ".\\data\\data\\masks"
images_names = os.listdir(images_path)

seed(123)
training_images_names = sample(images_names, int(len(images_names) * 0.7))
test_images_names = set(images_names) - set(training_images_names)

images_train = [load_image(os.path.join(images_path, name)) for name in training_images_names]
masks_train = [load_mask(os.path.join(masks_path, name[0:-3] + "png")) for name in training_images_names]
images_test = [load_image(os.path.join(images_path, name)) for name in test_images_names]
masks_test = [load_mask(os.path.join(masks_path, name[0:-3] + "png")) for name in test_images_names]

# Normalization
images_train = np.array(images_train) / 255
masks_train = np.array(masks_train) / 255
images_test = np.array(images_test) / 255
masks_test = np.array(masks_test) / 255

# Creating U-net model.
model_in = tf.keras.layers.Input((256, 256, 3))

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(model_in)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c1)
m1 = tf.keras.layers.MaxPool2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(m1)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c2)
m2 = tf.keras.layers.MaxPool2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(m2)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c3)

ct3 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c3)
con3 = tf.keras.layers.concatenate([ct3, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(con3)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c8)

ct4 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
con3 = tf.keras.layers.concatenate([ct4, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(con3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model([model_in], [outputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Training
history = model.fit(images_train, masks_train, batch_size=25, epochs=15)

# Create directory for generated images
if not os.path.exists(".\\examples"):
    os.makedirs(".\\examples")

# Save history data.
epochs = [i for i in range(len(history.history["accuracy"]))]
plt.plot(epochs, history.history["accuracy"])
plt.title("Accuracy over epochs")
plt.ylim([0, 1])
plt.savefig(".\\examples\\Accuracy_over_epochs.png")
plt.clf()

plt.plot(epochs, history.history["loss"])
plt.title("Loss over epochs")
plt.savefig(".\\examples\\Loss_over_epochs.png")

# Training IOU
iou_train = calculate_iou(model, images_train, masks_train)
print(f"IOU train: %.2f." % iou_train)

# Test IOU
iou_test = calculate_iou(model, images_test, masks_test)
print("IOU test: %.2f" % iou_test)

# Saving sample images and masks
save_samples(model, images_test[0:10], masks_test[0:10], ".\\examples")
