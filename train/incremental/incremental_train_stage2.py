import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# ✅ 設定參數
DATASET_DIR = "data/dataset/train"  # **包含新水果的爬蟲數據 + 30% 360 數據**
VALIDATION_DIR = "data/dataset/validation"
MODEL_SAVE_PATH = "models/fruit_classifier_updated.h5"  # **最終輸出**
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
INITIAL_LR = 0.00005  # **降低學習率**

# ✅ 載入「第一階段已訓練新水果爬蟲數據」的模型
base_model = load_model("models/fruit_classifier_new_stage1.h5")

# ✅ **解凍最後 20 層進行微調**
for layer in base_model.layers[-20:]:
    layer.trainable = True

# ✅ **重新編譯模型**
base_model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ 載入數據（新水果 + 30% 360 數據）
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ✅ 訓練模型
history = base_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
               EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True),
               ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1)],
)

# ✅ 儲存模型
base_model.save(MODEL_SAVE_PATH)
print(f"✅ 第二階段微調完成，模型已保存至 {MODEL_SAVE_PATH}")