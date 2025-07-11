import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# **📌 設定參數**
TRAIN_DIR = "data/dataset/train"      # 訓練數據目錄
VAL_DIR = "data/dataset/validation"   # 驗證數據目錄
MODEL_SAVE_PATH = "models/fruit_classifier_stage1.h5"  # **第一階段模型儲存路徑**
TARGET_SIZE = (224, 224)  # 影像尺寸
BATCH_SIZE = 32
EPOCHS = 30  # 訓練回合數
INITIAL_LR = 0.001  # 初始學習率

# **📌 確保 models 資料夾存在**
os.makedirs("models", exist_ok=True)

# **📌 標準化 (rescale)**
datagen = ImageDataGenerator(rescale=1.0 / 255)  

# **📌 載入數據**
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = datagen.flow_from_directory(  
    VAL_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# **📌 計算類別數量**
num_classes = len(train_generator.class_indices)
print(f"\n🛠 目前 Google 爬蟲的水果類別數量: {num_classes}")

# **📌 建立 MobileNetV2 模型**
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # **先凍結 MobileNetV2，避免破壞預訓練權重**

# **📌 自訂分類頭**
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax")(x)

# **📌 建立完整模型**
model = Model(inputs=base_model.input, outputs=predictions)

# **📌 編譯模型**
model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss="categorical_crossentropy", metrics=["accuracy"])

# **📌 設定回調函數**
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=8, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1)

# **📌 訓練模型**
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop, model_checkpoint]
)

# **📌 儲存模型**
model.save(MODEL_SAVE_PATH)
print(f"✅ 第一階段模型已保存至 {MODEL_SAVE_PATH}")

# **📌 繪製訓練過程**
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

plt.savefig("models/training_performance_stage1.png")
plt.show()

print("🎉 訓練完成！")