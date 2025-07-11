from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')

# USDA API 配置
USDA_API_KEY = "X2Kkx0le9B0U8ogZ1326uz1CgziXAeR7xynilHC1"
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# 動態加載類別名稱
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    'data/dataset/train',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)
classes = list(train_generator.class_indices.keys())

# 加載模型
model = load_model('models/fruit_classifier_updated.h5')

def get_usda_nutrition(fruit_name):
    """向 USDA API 查詢營養數據"""
    params = {
        "query": fruit_name,
        "api_key": USDA_API_KEY,
        "pageSize": 1
    }
    response = requests.get(USDA_BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        if "foods" in data and len(data["foods"]) > 0:
            nutrients = data["foods"][0].get("foodNutrients", [])
            nutrition_info = {}
            for nutrient in nutrients:
                name = nutrient.get("nutrientName")
                value = nutrient.get("value")
                unit = nutrient.get("unitName")
                if name and value:
                    nutrition_info[name] = f"{value} {unit}"
            return nutrition_info
    return {"message": "No nutrition data found."}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_file = request.files['image']
        img_path = os.path.join('static/uploads', img_file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img_file.save(img_path)

        # ✅ **正確的影像尺寸**
        img = load_img(img_path, target_size=(224, 224))  # **修正為 224x224**
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 預測
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]

        # **查詢 USDA API 營養資訊**
        nutrient_info = get_usda_nutrition(predicted_class)

        return jsonify({
            'prediction': predicted_class,
            'nutrients': nutrient_info
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=4000)
