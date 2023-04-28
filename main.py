import cv2
import time
import requests
import numpy as np
import tensorflow as tf
from datetime import datetime
from influxdb import InfluxDBClient

# Cargar modelo
model = tf.lite.Interpreter('./dig-cont_0611_s3.tflite')
model.allocate_tensors()

# Obtener los detalles de entrada y salida del modelo
detalles_entrada = model.get_input_details()[0]
detalles_salida = model.get_output_details()[0]
altura = detalles_entrada['shape'][1]
ancho = detalles_entrada['shape'][2]

cameras = [
    {
        'name': 'rincer',
        'ip': '172.16.150.10',
        'angle': 3,
        'x': 500, 'y': 150, 'w': 900, 'h': 180,
        'digits': 6
    },
    {
        'name': 'apenvasado',
        'ip': '172.16.150.11',
        'angle': 14,
        'x': 120, 'y': 165, 'w': 910, 'h': 200,
        'digits': 6
    }
]

# create a connection to the InfluxDB database
client = InfluxDBClient(host='localhost', port=8086, username='administrator', password='', database='papude')

while True:
    for camera in cameras:
        r = requests.get('http://' + camera['ip'] + '/photo', stream=True)
        if r.status_code == 200:
            #with open(photospath + camera['name'] + '.jpg', 'wb') as f:
            #    f.write(r.content)
            try:
                img = np.asarray(bytearray(r.raw.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                # Rotate the image
                rows, cols = img.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), camera['angle'], 1)
                img_rotated = cv2.warpAffine(img, M, (cols, rows))

                # Crop the image
                img = img_rotated[
                    camera['y']:camera['y']+camera['h'],
                    camera['x']:camera['x']+camera['w']
                ]
                w2 = round(camera['w'] / camera['digits'])
                digits = [None] * camera['digits']

                # Crop digits
                for digit in range(camera['digits']):
                    digits[digit] = img[0:camera['h'], w2*digit:w2*(digit+1)]

                value = ''
                # Read digits
                for digit in range(camera['digits']):
                    img = cv2.resize(digits[digit], (ancho, altura))
                    #img = img / 255.
                    #img = img.astype(np.float32)
                    img = img.astype("float32")
                    img = np.expand_dims(img, axis=0)
                    model.set_tensor(detalles_entrada['index'], img)
                    model.invoke()
                    digit = np.argmax(model.get_tensor(detalles_salida['index']))
                    value += str(digit)
                
                data = [
                    {
                        "measurement": 'caudal_' + camera['name'],
                        "tags": {
                            "location": "idk"
                        },
                        "time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "fields": {
                            "value": int(value)
                        }
                    }
                ]
                client.write_points(data)
            except Exception as e:
                print(e)
        else:
            print('Failed recieving image:', r.status_code)
            requests.get('http://' + camera['ip'] + '/reboot')
    
    time.sleep(300)