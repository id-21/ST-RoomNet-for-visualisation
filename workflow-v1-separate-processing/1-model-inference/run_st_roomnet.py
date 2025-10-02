import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
from xml.etree import ElementTree
import cv2
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add the shared directory to Python path
shared_dir = Path(__file__).parent.parent.parent / "shared"
print(shared_dir)
sys.path.insert(0, str(shared_dir))

from spatial_transformer_fixed import ProjectiveTransformerLayer
#from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import *
from scipy.interpolate import interp1d
import random
import scipy.io
import math
from scipy import ndimage
from sklearn.utils import shuffle

ABS_PATH_TO_IMG = "/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/images/room1.jpeg"
ABS_PATH_To_REF = "/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/images/ref_img2.png"
ABS_PATH_TO_WEIGHTS = "/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/Weight_ST_RroomNet_ConvNext.h5"

img = cv2.imread(ABS_PATH_TO_IMG)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400 , 400))
img1 =  np.array(img,copy=True)
img = img[tf.newaxis,...]
img = preprocess_input(img)

# seg = cv2.imread(val_path+'sun_atssgbmizunolhzn.png',0)
# seg = cv2.resize(seg, (400 ,400), interpolation= cv2.INTER_NEAREST)
# seg = seg/51.0
 
ref_img = tf.io.read_file(ABS_PATH_To_REF)
ref_img = tf.io.decode_png(ref_img)
ref_img = tf.cast(ref_img, tf.float32) / 51.0
ref_img = ref_img[tf.newaxis,...]
#ref_img = tf.tile(ref_img, [1,1,1,1])
print(ref_img.shape)


base_model = ConvNeXtTiny(include_top=False, weights="imagenet", input_shape= (400,400,3), pooling = 'avg')
theta = Dense(8)(base_model.output)
# stl= ProjectiveTransformer((400,400)).transform(ref_img, theta)
stl = ProjectiveTransformerLayer(ref_img, (400,400))(theta)
model = Model(base_model.input, stl)


model.summary()
model.load_weights(ABS_PATH_TO_WEIGHTS)


out= model.predict(img)

out = np.rint(out[0,:,:,0])
print("MODEL OUTPUT: ", out)
print("OUTPUT TYPE: ", type(out))

# Save full matrix without truncation
np.savetxt(f'outputs/st_roomnet_output.txt', out, fmt='%.1f',
            header=f'OUTPUT TYPE: {type(out)}\nOUTPUT SHAPE: {out.shape}\nMODEL OUTPUT:',
            comments='')

print("Output saved to st_roomnet_output.txt")

plt.figure('seg')
plt.imshow(out, vmin = 1, vmax= 5)
# plt.figure('gt')
# plt.imshow(seg , vmin = 1, vmax= 5)
plt.figure('img')
plt.imshow(img1)
plt.show()

