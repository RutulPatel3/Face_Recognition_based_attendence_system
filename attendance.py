import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path  = 'images'
img = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    img.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findEncodings(img):
    encodeList =[]
    for i in img:
        i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        dataList = f.readlines()
        nameList =[]
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now
            dtString = now().strftime('%H;%M;%S')
            f.writelines(f'\n{name}, {dtString}')

    


encodeListKnown = findEncodings(img)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, i = cap.read()
    iSmall = cv2.resize(i,(0,0),None,0.25,0.25)
    iSmall = cv2.cvtColor(iSmall,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(iSmall)
    encodeCurFrame = face_recognition.face_encodings(iSmall,facesCurFrame)

    for encodeface,faceloc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeface)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(i,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(i,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(i,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    
    cv2.imshow('webcam',i)
    cv2.waitKey(1)







"""
faceloc = face_recognition.face_locations(imgMine)[0]
encodeMine = face_recognition.face_encodings(imgMine)[0]
cv2.rectangle(imgMine,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeMine],encodeTest)
faceDis = face_recognition.face_distance([encodeMine],encodeTest)"""