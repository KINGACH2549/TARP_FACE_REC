# dict={'_id': 'deepface/tests/dataset/img2.jpg', 'distance': 7.0178008611285865, 'cond': True}
import collections
import pymongo

client=pymongo.MongoClient("mongodb://localhost:27017/")
print(client)
db=client["face_recognition"]
collection=db['employees']
for document in collection.find():
    print(document['embeddings'])
