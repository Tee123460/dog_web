from flask import Flask, render_template, request
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# โหลดโมเดล MobileNet
model = MobileNet(weights='imagenet')

# พจนานุกรมแปลชื่อสายพันธุ์ (จาก ImageNet ID → ภาษาไทย)
breed_th = {
    'n02113023': 'หมาบางแก้ว',     # pembroke
    'n02100735': 'หมาบางแก้ว',     # keeshond
    'n02109961': 'หมาบางแก้ว',     # eskimo_dog
    'n02111889': 'หมาบางแก้ว',     # samoyed
    'n02099601': 'โกลเด้น รีทรีฟเวอร์',
    'n02110958': 'ปั๊ก',
    'n02085620': 'ชิวาวา',
    'n02086910': 'ชิวาวา',         # toy_terrier
    'n02099712': 'ลาบราดอร์ รีทรีฟเวอร์',
    'n02110185': 'ไซบีเรียน ฮัสกี้',
    'n02088364': 'บีเกิล',
    'n02106662': 'เยอรมันเชฟเฟิร์ด',
    'n02107155': 'ดัลเมเชียน',
    'n02112018': 'ปอมเมอเรเนียน',
    'n02086240': 'ชิสุ',
    'n02086079': 'ทอยพุดเดิ้ล',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')
    results = []

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    for file in files:
        if file and file.filename:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            top5_raw = decode_predictions(preds, top=5)[0]

            top5_names = []
            for pred in top5_raw:
                imagenet_id = pred[0]
                name_th = breed_th.get(imagenet_id, pred[1])
                top5_names.append(name_th)

            results.append({
                'image_url': '/' + path,
                'top5': top5_names
            })

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
