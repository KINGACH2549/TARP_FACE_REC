from ast import Name
from modulefinder import STORE_NAME
from re import I
import tarfile
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

# def img_to_encoding(img,model):
#     img=cv2.resize(img,(160,160))
#     img=img.astype('float64')
#     img = np.around(np.array(img) / 255.0, decimals=12)
#     x_train = np.expand_dims(img, axis=0)
#     embedding = model.predict_on_batch(x_train)
#     return embedding / np.linalg.norm(embedding, ord=2)

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


client=pymongo.MongoClient("mongodb://localhost:27017/")
print(client)
db=client["face_recognition"]
collection=db["employees"]

# def query(target_embedding):
    
#     # collection=db['employees']
#     query = db.collection.aggregate( [
# {
#     "$addFields": { 
#         "target_embedding": target_embedding
#     }
# }
# , {"$unwind" : { "path" : "$embedding", "includeArrayIndex": "embedding_index"}}
# , {"$unwind" : { "path" : "$target_embedding", "includeArrayIndex": "target_index" }}
# , {
#     "$project": {
#         "Name": 1,
#         "embedding": 1,
#         "target_embedding": 1,
#         "compare": {
#             "$cmp": ['$embedding_index', '$target_index']
#         }
#     }
# }
# ,
#  {
#   "$group": {
#     "_id": "$Name",
#     "distance": {
#             "$sum": {
#                 "$pow": [{
#                     "$subtract": ['$embedding', '$target_embedding']
#                 }, 2]
#             }
#     }
#   }
# }
# , { 
#     "$project": {
#         "_id": 1,
#         # "distance": 1,
#         "distance": {"$sqrt": "$distance"}
#     }
# }
# , { 
#     "$project": {
#         "_id": 1
#         , "distance": 1
#         , "cond": { "$lte": [ "$distance", 10 ] }
#     }
# }
# , {"$match": {"cond": True}}
# , { "$sort" : { "distance" : 1 } }
# ] )
#     for i in query:
#         print(i)

def query(target_embedding):
    min_dist=100
    identity=0
    name="random"
    for document in collection.find():
         dist = np.linalg.norm(np.array(target_embedding)-np.array(document['embeddings']))
         if dist<min_dist:
            min_dist = dist
            identity = document['_id']
            name=document['Name']
    if min_dist>10:
        return 0,"CAN'T FIND !!"
    else:
         return identity,name


# while(True):
#     ret,frame=cam.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces=faceDetect.detectMultiScale(gray,1.3,5)
#     for(x,y,w,h) in faces:
#         # sampleNum=sampleNum+1
#         # cv2.imwrite("Images/"+str(name)+"."+id+'.'+ str(sampleNum) +".jpg",frame[y:y+h,x:x+w])
#         face_rec=frame[y:y+h,x:x+w]
#         target_embedding=img_to_encoding(face_rec,face_encoder)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
#         id,name=query(target_embedding.tolist())
#         # cv2.putText(frame,"ID: "+str(id), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#         # cv2.putText(frame, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
#         cv2.putText(frame, name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
#         # cv2.putText(frame,"Name: "+str(name), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     cv2.imshow('frame', frame)
#     # cv2.waitKey(100)
#     if(cv2.waitKey(10)==ord('q')):
#         break
# cam.release()
# cv2.destroyAllWindows()

sampleNum=0
while(True):
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("Images/test"+str(sampleNum) +".jpg",frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('frame', frame)
    cv2.waitKey(100)
    if(sampleNum>0):
        break
cam.release()
cv2.destroyAllWindows()
tar_emb=img_to_encoding("Images/test"+str(sampleNum) +".jpg",face_encoder)
id,name=query(tar_emb.tolist())

print(id)
print(name)
# for i in que:
#     print(i)
