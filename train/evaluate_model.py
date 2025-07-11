from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import shutil

# 配置參數（使用 argparse 獲取動態參數）
parser = argparse.ArgumentParser(description="Evaluate a trained fruit classification model.")
parser.add_argument('--model_path', type=str, default='models/fruit_classifier_new_stage1.h5', help="Path to the updated model file.")
parser.add_argument('--test_dir', type=str, default='data/dataset/test', help="Path to the test dataset.")
parser.add_argument('--output_dir', type=str, default='models', help="Directory to save evaluation results.")
parser.add_argument('--errors_dir', type=str, default='data/errors', help="Directory to save misclassified samples.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
parser.add_argument('--image_size', type=int, nargs=2, default=(100, 100), help="Image size for evaluation.")
parser.add_argument('--save_errors', action='store_true', help="Whether to save misclassified samples.")
args = parser.parse_args()

# 設置參數
MODEL_PATH = args.model_path
TEST_DIR = args.test_dir
OUTPUT_DIR = args.output_dir
ERRORS_DIR = args.errors_dir
BATCH_SIZE = args.batch_size
IMAGE_SIZE = tuple(args.image_size)

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
if args.save_errors:
    os.makedirs(ERRORS_DIR, exist_ok=True)

# 加載模型（改為 updated 模型）
print(f"加載模型: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# 測試數據生成器
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # 確保生成器不打亂數據順序
)

# 獲取真實標籤與預測結果
print("開始預測...")
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# 打印混淆矩陣與分類報告
print("\n混淆矩陣:")
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

print("\n分類報告:")
class_names = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

# 保存分類報告到文件
report_path = os.path.join(OUTPUT_DIR, 'classification_report_updated.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\n分類報告已保存至: {report_path}")

# 可視化混淆矩陣
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()

confusion_matrix_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_updated.png')
plot_confusion_matrix(cm, class_names, confusion_matrix_path)
print(f"\n混淆矩陣圖已保存至: {confusion_matrix_path}")

# 評估模型
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n測試集損失: {test_loss:.4f}")
print(f"測試集準確率: {test_accuracy:.2f}")

# 顯示分類錯誤的樣本，並保存到錯誤資料夾
incorrect_indices = np.where(y_true != y_pred_classes)[0]
if len(incorrect_indices) > 0:
    print(f"\n分類錯誤的樣本數量: {len(incorrect_indices)}")
    
    if args.save_errors:
        print(f"保存分類錯誤的樣本至 {ERRORS_DIR}...")
        for i in incorrect_indices:
            src_path = test_generator.filepaths[i]
            true_label = class_names[y_true[i]]
            pred_label = class_names[y_pred_classes[i]]
            
            # 在錯誤資料夾中為每個類別創建子資料夾
            error_class_dir = os.path.join(ERRORS_DIR, f"{true_label}_as_{pred_label}")
            os.makedirs(error_class_dir, exist_ok=True)
            
            # 保存錯誤影像
            shutil.copy(src_path, os.path.join(error_class_dir, os.path.basename(src_path)))
else:
    print("\n未發現分類錯誤的樣本！")