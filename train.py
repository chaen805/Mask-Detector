from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

train_dir = os.getcwd() + '/data/train'
test_dir = os.getcwd() + '/data/test'

# 하이퍼파라미터
INIT_LR = 1e-3
EPOCHS = 20
BS = 64

def load_data(data_path):
    imagePaths = list(paths.list_images(data_path))
    data = []
    labels = []
    
    data.clear()
    labels.clear()
    
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        
        image = load_img(imagePath, target_size=(50, 50))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(label)
        
    data = np.array(data, dtype='float32')
    labels = np.array(labels)
    
    # One-Hot Encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    
    return data, labels

print("loading images...")

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

print("number of train data :", len(X_train))
print("number of test data:", len(X_test))

# 데이터 나누기
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                 test_size = 0.2,
                                                 stratify=y_train,
                                                 random_state=42)

# 데이터 증강
aug = ImageDataGenerator(rotation_range=20,
                        zoom_range=0.15,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        horizontal_flip=True,
                        fill_mode="nearest")

# 모델 설계 및 학습
inputs = Input(shape=(50, 50, 3))

baseModel = MobileNetV2(include_top=False,
                        weights="imagenet", 
                        input_tensor=inputs)

x = baseModel.output
x = AveragePooling2D(pool_size=(1,1))(x)
x = Flatten(name="flatten")(x)
x = Dense(512, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

for layer in baseModel.layers:
    layer.trainable = False

model.summary()

print("compiling model...")

opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

H = model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_val, y_val),
    validation_steps=len(X_val) // BS,
    epochs=EPOCHS)

print("saving model...")

save_path = 'mask_detect_model.h5'
model.save(save_path)

N = EPOCHS

plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("loss_plot.png")

plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("acc_plot.png")

# 모델 평가
print("evaluating model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))