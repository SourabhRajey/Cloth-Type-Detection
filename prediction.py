# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import cv2
import glob
import os
import urllib.request
import numpy as np

from color_extractor import ImageToColor

npz = np.load('color_names.npz')
img_to_color = ImageToColor(npz['samples'], npz['labels'])
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]
model = load_model('my_model.h5')
def run_example(img_path,fcid):
	# load the image
	img = cv2.imread(img_path)
	cv2.imwrite('/home/saurabhraje/a.png',img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	obj={}
	obj["fcid"]=fcid
	_img_color = img_to_color.get(img)
	obj["color"]=_img_color[0] 
	img = load_image(img_path)
	# load model
	    
	# predict the class
	probs = model.predict(img)
	prediction = probs.argmax(axis=1)
	obj["type"]=labelNames[prediction[0]]
	return obj
        
fname='/home/saurabhraje/Desktop/a.txt'
with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
#print(content)	
j=0
for i in content:
	id1=i.split('/')
	id2=id1[7].split('.')
	fcid=id2[0]
	print(fcid)
	j+=1
	webp_imagepath="/home/saurabhraje/Desktop/firstcry/"+str(fcid)+".webp"
	#urllib.request.urlretrieve(i,webp_imagepath)
	png_path="/home/saurabhraje/Desktop/firstcry/"+str(fcid)+".png"
	os.system("curl "+i+" > "+webp_imagepath)
	os.system("dwebp "+webp_imagepath+" -o "+png_path)
	data=run_example(png_path,fcid)
	print(data)	
	with open('/home/saurabhraje/Desktop/ab.txt', 'a') as f:
		f.write(str(data)+"\n")
	f.close()
	os.system("rm "+webp_imagepath)
	os.system("rm "+png_path)
