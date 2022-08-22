from pprint import isrecursive
from random import sample
import cv2
import sqlite3
cam = cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def insert(Id,Name,age,gender,dept):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People;"
    cursor=conn.execute(cmd)
    data=[(id,str(Name),age,str(gender),str(dept))]
    cmd="INSERT INTO PEOPLE Values(?,?,?,?,?)"	
    conn.executemany(cmd,data) 
    conn.commit()
    conn.close()
id=input('Enter user id : ')
name=input('Enter your name : ')
age=input('Enter your age: ')
gender=input('Enter your gender: ')
dept=input('Enter your dept: ')
insert(id,name,age,gender,dept)


sampleNum=0
while(True):
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("Images/"+str(name)+"."+id+'.'+ str(sampleNum) +".jpg",frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('frame', frame)
    cv2.waitKey(100)
    if(sampleNum>2):
        break
cam.release()
cv2.destroyAllWindows()
