from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# classes for traffic signs
classes = {
    0 : 'Give way',
    1 : 'No entry',
    2 : 'One-way traffic',
    3 : 'One-way traffic',
    4 : 'No vehicles in both direction',
    5 : 'No entry for cycle',
    6 : 'No entry for goods vehicles',
    7 : 'No entry for pedestrians',
    8 : 'No entry for bullock carts',
    9 : 'No entry for hand carts',
    10 : 'No entry for motor vehicles',
    11 : 'Height limit',
    12 : 'Weight limit',
    13 : 'Axle weight limit',
    14 : 'Lenght limit',
    15 : 'No left turn',
    16 : 'No right turn',
    17 : 'No overtaking',
    18 : 'Maximum speed limit (90 km/h)',
    19 : 'Maximum speed limit (110 km/h)',
    20 : 'Horn prohibited',
    21 : 'No parking',
    22 : 'No stopping',
    23 : 'Turn left',
    24 : 'Turn right',
    25 : 'Steep descent',
    26 : 'Steep ascent',
    27 : 'Narrow Road',
    28 : 'Narrow bridge',
    29 : 'Unprotected quarry',
    30 : 'Road hump',
    31 : 'Dip',
    32 : 'Loose gravel',
    33 : 'Falling rocks',
    34 : 'Cattle',
    35 : 'Crossroads',
    36 : 'Side road junction',
    37 : 'Side road junction',
    38 : 'Oblique side road junction',
    39 : 'Oblique side road junction',
    40 : 'T-junction',
    41 : 'Y-junction',
    42 : 'Staggered side road junction',
    43 : 'Staggered side road junction',
    44 : 'Roundabout',
    45 : 'Guarded level crossing ahead',
    46 : 'Unguarded level crossing ahead',
    47 : 'Level crossing countdown maker',
    48 : 'Level crossing countdown Maker',
    49 : 'Level crossing countdown maker',
    50 : 'Level crossing countdown Maker',
    51 : 'Parking',
    52 : 'Bus stop',
    53 : 'First aid post',
    54 : 'Telephone',
    55 : 'Petrol pump / Filling station',
    56 : 'Hotel',
    57 : 'Restaurant',
    58 : 'Refreshments'
}

# predict the sign
def image_processing(img):
    model = load_model('./model/TSDR2.h5')
    data = []
    image = Image.open(img)
    image = image.convert("RGBA")
    r,g,b,a = image.split()
    image = Image.merge("RGB",(r,g,b))
    image = image.resize((60, 60))
    data.append(np.array(image))
    X_test = np.array(data)
    Y_pred = np.argmax(model.predict(X_test), axis=-1)
    return Y_pred

# main lending page
@app.route("/")
def index():
    return render_template('index.html')

# get sign from the user and pass to the model
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted Traffic🚦Sign is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)