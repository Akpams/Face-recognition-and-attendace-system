import cv2
import numpy as np
import face_recognition
import os
from tkinter import *
from datetime import datetime 
import pyrebase
##create a folder to hold the images to be recognized
# edit the image name to the name that will be recognized alongside with it.. 
path =r'C:\Users\Client\Documents\computer vision\from database\images\\'

#create a tkinter window to enable select gender before recognition
# gender = Tk()
# gender.title('GENDER')
# gender.geometry('200x200')
# mygender= ['MALE', 'FEMALE', 'OTHERS']
# options =StringVar()

# options.set('select gender')

# om1 = OptionMenu(gender, options, *mygender)
# choice=Variable.get(self=options)
# print(choice)
# om1.pack(expand=True)
# gender.mainloop()

##getting images from database

config ={
    'apiKey': "AIzaSyBF_zQsLersNz-jEIZqFEZZ4JhcGJSQdy2228",
    'authDomain': "face-recognition-2da61.fireba222seapp.com",
    'projectId': "face-recognition-2da61",
    'storageBucket': "face-recognition-2da61.appspot.com",
    'messagingSenderId': "204929907269",
    'appId': "1:204929907269:web:af60f3c7ec62650ed222a812c",
    'measurementId': "G-SWFLKS7JX9",
    'serviceAccount':'serviceAccount.json',
    'databaseURL': 'https://face-recognition-2da61-default-rtdb.firebaseio.com/'
  
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
path_cloud = 'images'

all_files = storage.child("images").list_files() # get all file
cnt = 0
#  print(all_files)
for file in all_files:
   print(file.name)
   names = file.name
   file.download_to_filename(path+names)
   cnt += 1


images =[]#list to hold images from the folder
className = [] #list to hold class name from the images 
for cls in os.listdir(path):#loop through for the image and class name from the folder
    currentImage = cv2.imread(f'{path}/{cls}')
    images.append(currentImage)
    className.append(os.path.splitext(cls)[0])
print(className)
print(len(images))
def findEncoding(images):#function to find encodings of images 
    encodingslist = []
    for img in images:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingslist.append(encode)
    return encodingslist
def attendance(name,choice):#function to write update to the csv file, the name, time, date and gender

    with open(r'C:\Users\Client\Documents\computer vision\attentence system\attendance.csv', 'r+') as f:
        datalist = f.readlines()
        namelist =[]
        for line in datalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            tString = now.strftime('%H:%M:%S')
            dString = now.strftime('%d:%m:%y')
            
            f.writelines(f'\n{name},{tString}, {choice}, {dString}')
                 
knownEncoding= findEncoding(images)
print('encoding complete')


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    img_cur_frame= cv2.resize(frame, (0,0), None, 0.25, 0.25)
    img_cur_frame = cv2.cvtColor(img_cur_frame, cv2.COLOR_BGR2RGB)
      
    face_cur_loc = face_recognition.face_locations(img_cur_frame)
    
    face_cur_enco = face_recognition.face_encodings(img_cur_frame, face_cur_loc)
    
    for encodeface, faceloc in zip(face_cur_enco,face_cur_loc):
        matches = face_recognition.compare_faces(knownEncoding,encodeface)
        faceDis = face_recognition.face_distance(knownEncoding,encodeface)
        # print(faceDis)
        matchidex = np.argmin(faceDis)
        if matches[matchidex]:
            name= className[matchidex].upper()
            # print(name)
            
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1=  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.rectangle(frame, (x1,y2-20),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
            attendance(name, choice)
            
            
            
    cv2.imshow('face recognition', frame)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
