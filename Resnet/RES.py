# %%
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


# %%
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import os

def preprocess_image(image, label, image_shape=224):
  image= tf.image.resize(image,[image_shape, image_shape])
  return tf.cast(image, tf.float32), label

# %%
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split = ["train", "validation"],
                                             shuffle_files = True,
                                             as_supervised = True,
                                             with_info = True)

# %%
ds_info

# %%
print(f"Training dataset size: {tf.data.experimental.cardinality(train_data).numpy()}")
print(f"Test dataset size: {tf.data.experimental.cardinality(test_data).numpy()}")

# %%
class_names = ds_info.features["label"].names
class_images = {}

for image, label in train_data:
    label_int = label.numpy()

    if label_int not in class_images:
        class_images[label_int] = image

    if len(class_images) == len(class_names):
        break

plt.figure(figsize=(20, 20))
for idx, (class_label, image) in enumerate(class_images.items()):
    plt.subplot(11, 10, idx + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(class_names[class_label])
    plt.axis("off")

plt.tight_layout()
plt.show()

# %%
train_data = train_data.map(map_func= preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = test_data.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

# %%
checkpoint_dir = "model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, "cp.weights2.h5")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      verbose=0)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=2,
                                                  restore_best_weights=True)

# %%
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
mixed_precision.global_policy()

# %%
input_shape = (224, 224, 3)

base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
base_model.trainable = False

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), # Use layers.RandomFlip directly
    layers.RandomRotation(0.2), # Use layers.RandomRotation directly
    layers.RandomZoom(0.2), # Use layers.RandomZoom directly
], name="data_augmentation")

inputs = layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(len(class_names))(x)
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])


print(f"Total layers in the base model: {len(base_model.layers)}")

model.summary()


# %%
history_feature_extraction = model.fit(train_data,
                                       epochs=2,
                                       steps_per_epoch=None,
                                       validation_data=test_data,
                                       validation_steps=None,
                                       callbacks=[model_checkpoint, early_stopping])

# %%
results_feature_extraction_model = model.evaluate(test_data)
results_feature_extraction_model

# %%
def calculate_metrics(model, test_data, class_names):
    y_true = []
    y_pred = []

    for images, labels in test_data:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(40, 40))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True, vmin=0, vmax=np.max(cm))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

calculate_metrics(model, test_data, class_names)

# %%
def plot_learning_curves(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_learning_curves(history_feature_extraction)

# %%
base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              metrics=["accuracy"])

print(f"Total layers in the base model: {len(base_model.layers)}")

trainable_layers = len([layer for layer in base_model.layers if layer.trainable])
non_trainable_layers = len([layer for layer in base_model.layers if not layer.trainable])

print(f"Number of trainable layers: {trainable_layers}")
print(f"Number of non-trainable layers: {non_trainable_layers}")

model.summary()

# %%
history_fine_tuning = model.fit(train_data,
                                epochs=2,
                                validation_data=test_data,
                                callbacks=[model_checkpoint, early_stopping])

# %%
results_feature_extraction_model = model.evaluate(test_data)
results_feature_extraction_model

# %%
def calculate_metrics(model, test_data, class_names):
    y_true = []
    y_pred = []

    for images, labels in test_data:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(40, 40))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True, vmin=0, vmax=np.max(cm))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

calculate_metrics(model, test_data, class_names)

# %%
def plot_learning_curves(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.tight_layout()
    plt.show()


plot_learning_curves(history_fine_tuning)



