from pprint import isrecursive
from random import sample
import cv2
import pymongo
# import sqlite3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL


from architecture import *
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)

cam = cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# def insert(Id,Name,age,gender,dept):
#     conn=sqlite3.connect("FaceBase.db")
#     cmd="SELECT * FROM People;"
#     cursor=conn.execute(cmd)
#     data=[(id,str(Name),age,str(gender),str(dept))]
#     cmd="INSERT INTO PEOPLE Values(?,?,?,?,?)"	
#     conn.executemany(cmd,data) 
#     conn.commit()
#     conn.close()
id=input('Enter user id : ')
name=input('Enter your name : ')
# age=input('Enter your age: ')
# # gender=input('Enter your gender: ')
# # # dept=input('Enter your dept: ')
# # # insert(id,name,age,gender,dept)

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def store(id,name,embd):
    client=pymongo.MongoClient("mongodb://localhost:27017/")
    print(client)
    db=client['face_recognition']
    collection=db['employees']
    dict={'_id':id,'Name':str(name),'embeddings':embd}
    collection.insert_one(dict)

sampleNum=0
while(True):
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("Images/"+str(name)+"."+id+'.'+ str(sampleNum) +".jpg",frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('frame', frame)
    cv2.waitKey(100)
    if(sampleNum>0):
        break
cam.release()
cv2.destroyAllWindows()



embd=img_to_encoding("Images/"+str(name)+"."+id+'.'+ str(sampleNum) +".jpg",face_encoder)
store(id,name,embd.tolist())

