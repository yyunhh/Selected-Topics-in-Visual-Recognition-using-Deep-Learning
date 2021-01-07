import os
import cv2
import pandas as pd
from PIL import Image

imagelist = os.listdir('./training_hr_images')

for i in imagelist:

    img_or = Image.open("./training_hr_images/" + i )
    (w, h) = img_or.size
    print(i)

    new_img = img_or.resize((int(w/3), int(h/3)),Image.BICUBIC)
    #new_img.show()
    new_img.save("./training_lr_images/"+ i  )