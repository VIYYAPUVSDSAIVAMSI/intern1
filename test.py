import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('damage.h5')
image=cv2.imread('')
img=Image.fromarray(image)

image.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)
result=model.prodict_classes(input_img)
print(int(result))
if result==0:
    model = load_model('type.h5')
    image = cv2.imread('')
    img = Image.fromarray(image)

    image.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    result = model.prodict_classes(input_img)
    print(result)
    if int(result)==0:
        print("type1")
    elif int(result)==1:
        print("type2")
    else:
        print("type3")
else:
    print("no damage")