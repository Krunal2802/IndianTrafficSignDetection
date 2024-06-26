import numpy as np
import os
import cv2

# input_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\41\\Reversed'
# output_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\41\\Reversed1'

input_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Reversed'
output_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Reversed1'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("dir is created!!!")

for file in os.listdir(input_path):
    filepath = os.path.join(input_path,file)
    original_image = cv2.imread(filepath)

    # unreversed Images Task
    ur_image = np.flipud(original_image)
    ur_image = cv2.resize(ur_image,(60,60))

    # save file
    output_filepath = os.path.join(output_path,file)
    cv2.imwrite(output_filepath,ur_image)