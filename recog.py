import collections
from ipaddress import collapse_addresses
import pymongo

if __name__=="__main__":
    client=pymongo.MongoClient("mongodb://localhost:27017/")
    print(client)
    db=client['first_time']
    collection=db['mysample']
    # dict={'Name':'harry','Age':21}
    # collection.insert_one(dict)
    dict={'Name':'achintya'}
    collection.insert_one(dict)
    


