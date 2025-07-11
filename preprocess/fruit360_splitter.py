import os
import random
import shutil

# ✅ 設定資料夾
FRUIT360_DIR = "data/fruit360"
DATASET_TRAIN_DIR = "data/dataset/train"
DATASET_VAL_DIR = "data/dataset/validation"

# ✅ 只取 30% 的 Fruit 360 數據
RATIO = 0.3
VAL_RATIO = 0.2  # 其中 20% 用作驗證集

# ✅ 取得現有的水果類別（來自 Google 爬蟲數據）
existing_classes = {cls for cls in os.listdir(DATASET_TRAIN_DIR) if not cls.startswith(".")}

# ✅ 統計影像數量
total_added_train = 0
total_added_val = 0

for fruit in os.listdir(FRUIT360_DIR):
    fruit_source_dir = os.path.join(FRUIT360_DIR, fruit)

    # ✅ **跳過隱藏檔案**
    if not os.path.isdir(fruit_source_dir):
        continue

    # ✅ **檢查 `train/` 內是否已經有這個水果類別**
    fruit_target_dir_train = os.path.join(DATASET_TRAIN_DIR, fruit)
    fruit_target_dir_val = os.path.join(DATASET_VAL_DIR, fruit)

    # ✅ **如果 `train/` 內已經有 360 影像，則跳過**
    existing_train_images = [img for img in os.listdir(fruit_target_dir_train) if img.startswith("360_")]
    if len(existing_train_images) > 0:
        print(f"🚀 `{fruit}` 在 `train/` 內已經有 360 影像，跳過！")
        continue  # ✅ **已經有 360 影像，則不再添加**

    # ✅ 確保目標資料夾存在
    os.makedirs(fruit_target_dir_train, exist_ok=True)
    os.makedirs(fruit_target_dir_val, exist_ok=True)

    # ✅ 取得 Fruit 360 影像清單
    images = [img for img in os.listdir(fruit_source_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]

    if len(images) == 0:
        print(f"⚠️ 類別 `{fruit}` 在 Fruit 360 無可用影像，跳過！")
        continue

    # ✅ **確保影像數量充足**
    num_to_add = min(int(len(images) * RATIO), len(images))  # **確保不超過現有數量**
    selected_images = random.sample(images, num_to_add)

    for img_file in selected_images:
        train_or_val = random.random()  # 生成隨機數，決定影像是放入訓練集還是驗證集
        src_path = os.path.join(fruit_source_dir, img_file)

        # ✅ **確保新影像不會因為重複名稱而被跳過**
        img_new_name = f"360_{img_file}"  # **加上前綴，避免與爬蟲影像衝突**
        
        if train_or_val < VAL_RATIO:
            target_path = os.path.join(fruit_target_dir_val, img_new_name)
            total_added_val += 1
        else:
            target_path = os.path.join(fruit_target_dir_train, img_new_name)
            total_added_train += 1

        shutil.copy(src_path, target_path)

print("\n✅ 已成功加入 Fruit 360 影像到 dataset！")
print(f"📊 加入至訓練集: {total_added_train} 張圖片")
print(f"📊 加入至驗證集: {total_added_val} 張圖片")
print("🔒 數據處理程序結束！")