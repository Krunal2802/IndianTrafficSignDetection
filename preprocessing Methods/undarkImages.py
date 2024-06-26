import numpy as np
import os
import cv2
from PIL import Image

# input_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\42\\Dark'
# output_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\42\\Dark1'

input_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Dark'
output_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Dark1'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("dir is created!!!")

for file in os.listdir(input_path):
    filepath = os.path.join(input_path,file)
    original_image = cv2.imread(filepath)

    alpha = 3 # Contrast control
    beta = 10 # Brightness control
    
    ud_image = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)

    output_filepath = os.path.join(output_path,file)
    cv2.imwrite(output_filepath,ud_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
