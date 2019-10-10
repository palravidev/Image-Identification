# Import Libraries
import numpy as np
import json
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input

#Upload VGG16
vgg16 = VGG16(weights='imagenet')
#Save
vgg16.save('vgg16.h5')

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
    
y = prepare_image('C:/Users/Public/data1a/predict/7.jpg')

preds = vgg16.predict(y)

preds.shape

CLASS_INDEX = None

CLASS_INDEX_PATH = 'C:/Users/Public/data1a/jsonfile/imagenet_class_index.json'
#CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def get_predictions(preds,top=5):
    global CLASS_INDEX
    
    #Load the jason file
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))
    
    #get the results
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results
    
print (get_predictions(preds, top=5))

#First Check - category list creation
from collections import Counter, defaultdict
import os
import pickle as pk

def get_car_categories():
    d = defaultdict(float)
    img_list = os.listdir('C:/Users/Public/data1a/training/01-whole')
    for i, img_path in enumerate(img_list):
        img = prepare_image('C:/Users/Public/data1a/training/01-whole/'+img_path)
        out = vgg16.predict(img)
        top = get_predictions(out, top=5)
        for j in top[0]:
            d[j[0:2]] += j[2]
        if i % 100 == 0:
            print (i, '/', len(img_list), 'complete')
    return Counter(d)

cat_counter=get_car_categories()

cat_list  = [k for k, v in cat_counter.most_common()[:30]]
cat_list

def car_categories_get(image_path, cat_list):
    img = prepare_image(image_path)
    out = vgg16.predict(img)
    top = get_predictions(out, top=5)
    print ("Validating that this is a picture of your car...")
    for j in top[0]:
        if j[0:2] in cat_list:
            print (j[0:2])
            return "Validation complete - proceed to damage evaluation"
    return "Are you sure this is a picture of your car? Please take another picture (try a different angle or lighting) and try again."

car_categories_get('C:/Users/Public/2020.jpeg', cat_list)
