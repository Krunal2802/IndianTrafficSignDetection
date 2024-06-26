import numpy as np
import os
import cv2

# input_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\40\\Reversed Mirrored'
# output_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\40\\Reversed Mirrored1'
# output_path1 = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\0\\Reversed Mirrored2'

input_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Reversed Mirrored'
output_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Reversed Mirrored1'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("dir is created!!!")

for file in os.listdir(input_path):
    filepath = os.path.join(input_path,file)
    original_image = cv2.imread(filepath)
    rows, cols, channels = original_image.shape
    urm_image = np.zeros((rows,cols,channels), dtype = np.uint8)
    for i in range(rows):
        for j in range(cols):
            urm_image[i, j] = original_image[rows-1-i, cols-1-j]

    # save file
    urm_image = cv2.resize(urm_image,(60,60))
    output_filepath = os.path.join(output_path,file)
    cv2.imwrite(output_filepath,urm_image)