import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# ✅ 確保 TensorFlow 運行環境穩定
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ 設定參數
DATASET_DIR = "data/dataset/train"  # 只包含新水果的爬蟲數據
VALIDATION_DIR = "data/dataset/validation"
MODEL_SAVE_PATH = "models/fruit_classifier_new_stage1.h5"  # **第一階段輸出**
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # ✅ **減少 Epoch，避免破壞舊知識**
INITIAL_LR = 0.0001  # ✅ **保持低學習率，避免舊類別權重大幅變動**

# ✅ 載入舊模型（已包含舊水果 + 360 數據）
base_model = load_model("models/fruit_classifier_final.h5")

# ✅ 取得新水果的類別
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
new_num_classes = len(train_generator.class_indices)

# ✅ 取得舊類別數量
old_num_classes = base_model.output_shape[-1]

# ✅ **檢查是否有新類別**
if new_num_classes > old_num_classes:
    print(f"🔄 檢測到新水果，從 {old_num_classes} 類 ➝ {new_num_classes} 類，更新分類層！")

    # ✅ **凍結舊層，避免破壞舊水果的學習**
    for layer in base_model.layers:
        layer.trainable = False  # **舊類別的權重不動**

    # ✅ **取出舊模型的分類層**
    x = base_model.layers[-2].output
    new_output = Dense(new_num_classes, activation="softmax", name="new_output_layer")(x)  # **確保名稱唯一**
    updated_model = Model(inputs=base_model.input, outputs=new_output)

    # ✅ **初始化分類層權重**
    old_weights = base_model.layers[-1].get_weights()
    new_weights_w = np.random.normal(size=(old_weights[0].shape[0], new_num_classes)) * 0.01
    new_weights_b = np.zeros(new_num_classes)
    new_weights_w[:, :old_num_classes] = old_weights[0]  # **保留舊類別權重**
    new_weights_b[:old_num_classes] = old_weights[1]

    updated_model.layers[-1].set_weights([new_weights_w, new_weights_b])
else:
    print("✅ 沒有新類別，直接使用原模型")
    updated_model = base_model

# ✅ **重新編譯模型**
updated_model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ 設定驗證集
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
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

# ✅ 訓練模型（只訓練新水果）
history = updated_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop, model_checkpoint],
)

# ✅ 儲存第一階段模型
updated_model.save(MODEL_SAVE_PATH)
print(f"✅ 第一階段模型已保存至 {MODEL_SAVE_PATH}")

# ✅ **畫出訓練結果**
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

plt.savefig("models/training_performance_new_stage1.png")
plt.show()

print("🎉 第一階段訓練完成（只訓練新水果的爬蟲數據）！")