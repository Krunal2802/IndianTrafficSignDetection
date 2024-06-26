import numpy as np
import os
import cv2

# input_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\41\\Normal'
# output_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\41\\Normal1'

input_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Normal'
output_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Normal1'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("dir is created!!!")

for file in os.listdir(input_path):
    filepath = os.path.join(input_path,file)
    original_image = cv2.imread(filepath)

    # resize the images
    resize_image = cv2.resize(original_image,(60,60))
    output_filepath = os.path.join(output_path,file)
    cv2.imwrite(output_filepath,resize_image)