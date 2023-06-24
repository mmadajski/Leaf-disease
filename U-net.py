import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from random import sample, seed
import matplotlib.pyplot as plt

# Read and prepare data.
path = os.getcwd()
images_path = os.path.join(path, "data\\data\\images")
masks_path = os.path.join(path, "data\\data\\masks")
images_names = os.listdir(images_path)

seed(123)
training_images_names = sample(images_names, int(len(images_names) * 0.7))
test_images_names = set(images_names) - set(training_images_names)
images_train = []
masks_train = []
images_test = []
masks_test = []

for name in training_images_names:
    image = cv2.imread(os.path.join(images_path, name))
    image_resize = cv2.resize(image, [256, 256])
    images_train.append(image_resize)

    mask_name = name[0:-3] + "png"
    mask = cv2.split(cv2.imread(os.path.join(masks_path, mask_name)))[2]
    mask_resize = cv2.resize(mask, [256, 256])
    mask_bin = np.where((mask_resize > 64), 255, 0)
    masks_train.append(mask_bin)

for name in test_images_names:
    image = cv2.imread(os.path.join(images_path, name))
    image_resize = cv2.resize(image, [256, 256])
    images_test.append(image_resize)

    mask_name = name[0:-3] + "png"
    mask = cv2.split(cv2.imread(os.path.join(masks_path, mask_name)))[2]
    mask_resize = cv2.resize(mask, [256, 256])
    mask_bin = np.where((mask_resize > 64), 255, 0)
    masks_test.append(mask_bin)

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
history = model.fit(images_train, masks_train, batch_size=25, epochs=50)

# Save history data.
epochs = [i for i in range(len(history.history["accuracy"]))]
plt.plot(epochs, history.history["accuracy"])
plt.savefig("accuracy.png")
plt.clf()

plt.plot(epochs, history.history["loss"])
plt.savefig("loss.png")

calc_iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
train_pred = model.predict(images_train)
prob_to_class = tf.map_fn(fn=lambda x: float(int(x > 0.5)), elems=train_pred)
IOU_train = calc_iou(prob_to_class, masks_train)
print(f"IOU train: %.2f." % IOU_train)

test_pred = model.predict(images_test)
test_prob_to_class = tf.map_fn(fn=lambda x: float(int(x > 0.5)), elems=test_pred)
IOU_test = calc_iou(test_prob_to_class, masks_test)
print("IOU test: %.2f" % IOU_test)
test_prob_to_class_np = tf.get_static_value(test_prob_to_class)

# Saving sample images and masks
for i in range(10):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Sample image, prediction and mask.")
    image = (images_test[i] * 255).astype("uint8")
    pred_mask = (test_prob_to_class_np[i] * 255).astype("uint8")
    mask = (masks_test[i] * 255).astype("uint8")

    ax1.imshow(image)
    ax2.imshow(pred_mask, cmap="Greys")
    ax3.imshow(mask, cmap="Greys")

    plt.savefig("Sample_" + str(i) + ".png")
