import matplotlib.pyplot as plt
import tensorflow as tf
import urllib.request
import zipfile
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_url = "https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip"
urllib.request.urlretrieve(data_url, "rps.zip")
local_file = "rps.zip"
zip_ref = zipfile.ZipFile(local_file, "r")
zip_ref.extractall("data/")
zip_ref.close()

main_path = "data/rps/"

# Data Augmentation
train_data = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, validation_split=0.2
)

# Data Flow
train_gen = train_data.flow_from_directory(
    main_path, target_size=(150, 150), class_mode="categorical", subset="training"
)

test_gen = train_data.flow_from_directory(
    main_path, target_size=(150, 150), class_mode="categorical", subset="validation"
)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(units=3, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_gen, validation_data=test_gen, epochs=2, verbose=1, callbacks=[checkpoint]
)

train_evaluation = model.evaluate(train_gen, verbose=0)
validation_evaluation = model.evaluate(test_gen, verbose=0)

# Output Section
with open("results.txt", "w") as outfile:
    lt, at = "loss : "+str(train_evaluation[0]), "accuracy : "+str(train_evaluation[1])
    lv, av = "loss : "+str(validation_evaluation[0]), "accuracy : "+str(validation_evaluation[1])
    outfile.write("Train Evaluation")
    output.write(lt)
    output.write(at)
    outfile.write("Validation Evaluation")
    output.write(lv)
    output.write(av)

loss, accuracy = history.history["loss"], history.history["accuracy"]
val_loss, val_accuracy = history.history["val_loss"], history.history["val_accuracy"]
epochs = range(len(history.history["loss"]))


# Plot loss
plt.style.use("seaborn-whitegrid")
plt.plot(epochs, loss, label="training_loss")
plt.plot(epochs, val_loss, label="val_loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("loss.png")

# Plot accuracy
plt.figure()
plt.plot(epochs, accuracy, label="training_accuracy")
plt.plot(epochs, val_accuracy, label="val_accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("accuracy.png")

plt.close()
