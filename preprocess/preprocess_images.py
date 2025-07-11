import os
import logging
import numpy as np
from PIL import Image, UnidentifiedImageError
from keras.preprocessing.image import ImageDataGenerator

# ✅ 設定參數
RAW_IMAGES_DIR = "data/raw_images"      
PROCESSED_DIR = "data/processed_images"  
IMAGE_SIZE = (224, 224)  
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg")  

# ✅ 數據增強
augmentor = ImageDataGenerator(
    rotation_range=20,        
    width_shift_range=0.1,    
    height_shift_range=0.1,   
    zoom_range=0.1,           
    horizontal_flip=True,    
    fill_mode='nearest'
)

# ✅ 設置日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_images(raw_dir, processed_dir, image_size):
    """🚀 處理影像，優化資料夾層級判斷"""
    success_count, error_count = 0, 0

    for fruit in os.listdir(raw_dir):
        fruit_dir = os.path.join(raw_dir, fruit)
        processed_fruit_dir = os.path.join(processed_dir, fruit)

        # ✅ 跳過非資料夾
        if not os.path.isdir(fruit_dir):
            continue

        # ✅ 若資料夾已處理過，則直接跳過
        if os.path.exists(processed_fruit_dir) and os.listdir(processed_fruit_dir):
            logging.info(f"🔄 資料夾已處理過，跳過: {fruit}")
            continue

        os.makedirs(processed_fruit_dir, exist_ok=True)
        logging.info(f"🆕 開始處理: {fruit}")

        image_files = [f for f in os.listdir(fruit_dir) if f.lower().endswith(SUPPORTED_FORMATS)]

        for img_file in image_files:
            img_path = os.path.join(fruit_dir, img_file)
            output_path = os.path.join(processed_fruit_dir, img_file)

            try:
                # ✅ 讀取影像
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)

                # ✅ 轉換為 NumPy 陣列，進行數據增強
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # ✅ 產生 3 張增強影像
                gen = augmentor.flow(img_array, batch_size=1, save_to_dir=processed_fruit_dir, save_format="jpeg")
                for _ in range(3):  
                    gen.next()

                # ✅ 儲存原始影像
                img.save(output_path, "JPEG")
                success_count += 1

            except UnidentifiedImageError:
                logging.error(f"❌ 無法識別影像: {img_path}")
                error_count += 1
            except Exception as e:
                logging.error(f"⚠️ 影像處理錯誤 {img_path}: {e}")
                error_count += 1

    logging.info(f"🎉 影像處理完成！成功: {success_count}, 錯誤: {error_count}")

if __name__ == "__main__":
    preprocess_images(RAW_IMAGES_DIR, PROCESSED_DIR, IMAGE_SIZE)

logging.info("🔒 影像處理程序已結束！")