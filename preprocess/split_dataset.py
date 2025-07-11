import os
import shutil
from sklearn.model_selection import train_test_split

# ✅ 定義路徑
PROCESSED_DIR = "data/processed_images"
DATASET_DIR = "data/dataset"

# ✅ 測試集與驗證集比例
TEST_SIZE = 0.2  
VALIDATION_SIZE = 0.2  

# ✅ 支援的影像格式
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")

# ✅ **確保 `train/validation/test` 目錄存在**
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(DATASET_DIR, split), exist_ok=True)

# ✅ 取得 **已經分割過的水果類別**
existing_classes = set(os.listdir(os.path.join(DATASET_DIR, "train")))

# ✅ 統計處理的影像數量
total_count = {"train": 0, "validation": 0, "test": 0}

for fruit in os.listdir(PROCESSED_DIR):
    fruit_dir = os.path.join(PROCESSED_DIR, fruit)

    # ✅ 跳過非資料夾
    if not os.path.isdir(fruit_dir):
        continue

    # ✅ **檢查該水果是否已經存在於 train/validation/test**
    fruit_train_dir = os.path.join(DATASET_DIR, "train", fruit)
    fruit_val_dir = os.path.join(DATASET_DIR, "validation", fruit)
    fruit_test_dir = os.path.join(DATASET_DIR, "test", fruit)

    if os.path.exists(fruit_train_dir) and len(os.listdir(fruit_train_dir)) > 0:
        print(f"🔄 水果 `{fruit}` 已經存在於資料集，跳過分割！")
        continue  # **跳過已分割過的水果**

    # ✅ **新水果，執行分割**
    print(f"🆕 新增水果 `{fruit}`，開始分割...")

    # ✅ 確保 `train/validation/test` 目錄存在
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(DATASET_DIR, split, fruit), exist_ok=True)

    # ✅ 取得該水果的影像
    images = [img for img in os.listdir(fruit_dir) if img.lower().endswith(SUPPORTED_FORMATS)]
    
    if len(images) == 0:
        print(f"[⚠️] {fruit} 沒有可用影像，跳過。")
        continue

    # ✅ 分割數據
    train, test = train_test_split(images, test_size=TEST_SIZE, random_state=42)
    train, val = train_test_split(train, test_size=VALIDATION_SIZE, random_state=42)

    # ✅ **複製影像到對應資料夾**
    for split, split_images in zip(["train", "validation", "test"], [train, val, test]):
        split_dir = os.path.join(DATASET_DIR, split, fruit)

        for img_file in split_images:
            src_path = os.path.join(fruit_dir, img_file)
            dst_path = os.path.join(split_dir, img_file)

            if not os.path.exists(dst_path):  # **避免重複複製**
                shutil.copy(src_path, dst_path)
                total_count[split] += 1

print(f"\n🎉 數據分割完成！")
print(f"📊 訓練集新增: {total_count['train']} 張圖片")
print(f"📊 驗證集新增: {total_count['validation']} 張圖片")
print(f"📊 測試集新增: {total_count['test']} 張圖片")
print("🔒 數據處理程序已結束！")