import numpy as np
import os
import cv2

# input_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\Database\\38'
# output_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\Database\\39'

# input_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\40\\Mirrored'
# output_path = 'C:\\Users\\Dell\\OneDrive\\Desktop\\40-58\\40\\Mirrored1'

input_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Mirrored'
output_path = 'D:\\TrafficSignDetectionSystem\\Train\\Images\\46\\Mirrored1'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("dir is created!!!")

for file in os.listdir(input_path):
    filepath = os.path.join(input_path,file)
    original_image = cv2.imread(filepath)

    # unmirrored Image Task
    um_image = np.fliplr(original_image)
    um_image = cv2.resize(um_image,(60,60))

    # save file
    output_filepath = os.path.join(output_path,file)
    cv2.imwrite(output_filepath,um_image)