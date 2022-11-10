import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os
import time
from pathlib import Path
import random
import subprocess
from tkinter import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'




# Create the model
model = Sequential() #use of Sequential model

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax')) #softmax function


model.load_weights('model.h5')

print('\n Welcome to Music Player based on Facial Emotion Recognition \n')
print('\n Press \'q\' to exit the music player \n')
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#File to append the emotions
with open(str(Path.cwd())+"\emotion.txt","w") as emotion_file:
                
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    now = time.time()  ###For calculate seconds of video
    future = now + 5
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            text = emotion_dict[maxindex]
            print(text)
            emotion_file.write(emotion_dict[maxindex]+"\n")
            emotion_file.flush()

        cv2.imshow('Video', cv2.resize(frame,(1220,900),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

        if time.time() > future:  ##after 5 second music will play
            cv2.destroyAllWindows()
            mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe' # copy your windows media player address
            if text == 'Angry':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Angry"))
                print('You are Angry......,I will play song for you :' + randomfile)
                file = ('E:\Anwar\Projects\Anwar Emotion Music Player\songs\Angry/' + randomfile) # cpoy your directory address
                subprocess.call([mp, file])

            if text == 'Disgusted':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Disgusted"))
                print('You are Disgusted......,I will play song for you :' + randomfile)
                file = ('E:\Anwar\Projects\Anwar Emotion Music Player\songs\Disgusted/' + randomfile) # cpoy your directory address
                subprocess.call([mp, file])

            if text == 'Fearful':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Fearful"))
                print('You are Fearful......,I will play song for you :' + randomfile)
                file = ("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Fearful/" + randomfile) # cpoy your directory address
                subprocess.call([mp, file])

            if text == 'Happy':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Happy"))
                print('You are Happy......,I will play song for you :' + randomfile)
                file = ('E:\Anwar\Projects\Anwar Emotion Music Player\songs\Happy/' + randomfile) # cpoy your directory address
                subprocess.call([mp, file])
            
            if text == 'Neutral':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs/Neutral"))
                print('You are Neutral......,I will play song for you :' + randomfile)
                file = ('E:\Anwar\Projects\Anwar Emotion Music Player\songs/Neutral/' + randomfile) # cpoy your directory address
                subprocess.call([mp, file])

            if text == 'Sad':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Sad"))
                print('You are Sad......,I will play song for you :' + randomfile)
                file = ('E:\Anwar\Projects\Anwar Emotion Music Player\songs\Sad/' + randomfile) # cpoy your directory address
                subprocess.call([mp, file])

            if text == 'Surprised':
                randomfile = random.choice(os.listdir("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Surprised"))
                print('You are Surprised......,I will play song for you :' + randomfile)
                file = ("E:\Anwar\Projects\Anwar Emotion Music Player\songs\Surprised/" + randomfile) # cpoy your directory address
                subprocess.call([mp, file])

            print(randomfile)

            future = time.time() + 5
        
    cap.release()
        