import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# ✅ 設定參數
DATASET_DIR = "data/dataset/train"
VALIDATION_DIR = "data/dataset/validation"
MODEL_SAVE_PATH = "models/fruit_classifier_final2.h5"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  # 微調不需要太多 Epoch
INITIAL_LR = 0.0001  # **降低學習率**

# ✅ 載入 Stage 1 模型
base_model = load_model("models/fruit_classifier_stage1.h5")

# ✅ 解凍 MobileNetV2 的最後 50 層進行微調
for layer in base_model.layers[-50:]:
    layer.trainable = True

# ✅ 重新編譯模型（確保調整學習率）
base_model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ **移除數據增強，只做 rescale**
train_datagen = ImageDataGenerator(rescale=1.0 / 255)  
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ✅ 載入數據
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ✅ 設定回調函數
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1)

# ✅ 訓練模型
history = base_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop, model_checkpoint],
    verbose=2
)

# ✅ 儲存模型
base_model.save(MODEL_SAVE_PATH)
print(f"✅ 第二階段微調完成，模型已保存至 {MODEL_SAVE_PATH}")

# ✅ 繪製訓練過程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.savefig("models/training_performance_final.png")
plt.show()

print("🎉 微調完成！")