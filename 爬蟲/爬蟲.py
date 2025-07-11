import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import requests
from io import BytesIO

# 設定下載資料夾與搜尋關鍵字
Fruit_Name = "Mandarine"        
SEARCH_TERM = f"real {Fruit_Name} images high quality isolated"
SAVE_DIR = "/Users/tim/Desktop/蔬果AI/使用的影像數據/{}".format(Fruit_Name)
NUM_IMAGES = 1000  # 目標下載數量
MIN_WIDTH = 50  # 最小圖像寬度
MIN_HEIGHT = 50  # 最小圖像高度
SCROLL_TIMES = 200  # 滾動次數

os.makedirs(SAVE_DIR, exist_ok=True)

# 啟動 Chrome 瀏覽器
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 無頭模式（不開視窗）
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 打開 Google 圖片
search_url = f"https://www.google.com/search?tbm=isch&q={SEARCH_TERM}"
driver.get(search_url)
time.sleep(3)

# 滾動頁面以加載更多圖片
for _ in range(SCROLL_TIMES):
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

# 嘗試點擊「顯示更多結果」按鈕（多次嘗試）
for _ in range(10):
    try:
        more_button = driver.find_element(By.XPATH, "//input[@value='Show more results']")
        more_button.click()
        time.sleep(5)  # 等待加載
    except Exception:
        break  # 如果沒有按鈕，就跳出迴圈

# 找到所有圖片標籤
images = driver.find_elements(By.CSS_SELECTOR, "img")

count = 0
for i, img in enumerate(images):
    try:
        src = img.get_attribute("src")

        # 過濾掉 base64 編碼的圖片
        if not src or "data:image" in src:
            print(f"[跳過] 無效圖片: {src}")
            continue

        # 下載圖片
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(src, headers=headers, timeout=10)
        image = Image.open(BytesIO(response.content))

        # 確保圖片解析度符合要求
        if image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
            print(f"[跳過] 圖像太小: {image.width}x{image.height}")
            continue

        # 儲存圖片，使用統一命名格式
        filename = os.path.join(SAVE_DIR, f"{Fruit_Name}{count+1:03d}.jpg")
        image.save(filename, "JPEG")

        count += 1
        print(f"[下載] {filename} ({image.width}x{image.height})")

        if count >= NUM_IMAGES:
            break

    except Exception as e:
        print(f"[錯誤] 無法下載圖片 {i}: {e}")

driver.quit()
print(f"下載完成，共獲取 {count} 張圖片！")
