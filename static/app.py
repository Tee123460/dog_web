from flask import Flask, render_template, request
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = MobileNet(weights='imagenet')

dog_breeds = {
    'n02113023': 'หมาบางแก้ว',              # Pembroke Welsh Corgi
    'n02099601': 'โกลเด้น รีทรีฟเวอร์',     # golden retriever
    'n02110958': 'ปั๊ก',                   # pug
    'n02085620': 'ชิวาวา',                # chihuahua
    'n02099712': 'ลาบราดอร์ รีทรีฟเวอร์', # labrador retriever
    'n02110185': 'ไซบีเรียน ฮัสกี้',       # siberian husky
    'n02088364': 'บีเกิล',                 # beagle
    'n02106662': 'เยอรมันเชฟเฟิร์ด',       # german shepherd
    'n02107155': 'ดัลเมเชียน',             # dalmatian
    'n02112018': 'ปอมเมอเรเนียน',          # pomeranian
    'n02086240': 'ชิสุ',                    # shih-tzu
    'n02086079': 'ทอยพุดเดิ้ล',             # toy poodle
    'n02111889': 'หมาบางแก้ว',              # samoyed
    'n02109961': 'หมาบางแก้ว',              # eskimo dog
    'n02100735': 'หมาบางแก้ว',              # keeshond
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]  # top 5 predictions

    # เตรียม list ของ top5 แปลชื่อไทยถ้ามี ไม่งั้นแสดงชื่ออังกฤษ
    top5 = []
    for pred in decoded:
        code, eng_name, score = pred
        name_th = dog_breeds.get(code, eng_name)
        top5.append(f"{name_th} ({eng_name}) - {score*100:.2f}%")

    # เอาอันดับ 1 มาโชว์เป็น result หลัก (แปลไทยด้วย)
    result_th = dog_breeds.get(decoded[0][0], decoded[0][1])

    return render_template('index.html', result=result_th, top5=top5, image_url='/' + file_path)

if __name__ == '__main__':
    app.run(debug=True)
