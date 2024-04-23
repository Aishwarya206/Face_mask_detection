from flask import Flask, render_template, request, Response
import sqlite3
import re
import base64
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

model = load_model('face_mask_detection_model.keras')
protoPath = "CaffeeModel/deploy.prototxt.txt"
modelPath= "CaffeeModel/res10_300x300_ssd_iter_140000.caffemodel"
cvNet = cv2.dnn.readNetFromCaffe(protoPath,modelPath)

def capture_image_on_person():
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Image Capturer App")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture the pretty face")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Image App", frame)

        if len(faces) > 0:  # Capture image when at least one face is detected
            img_name = "opencv_frame.png"
            cv2.imwrite(img_name, frame)
            print("Screenshot taken: {}".format(img_name))
            # You can choose to return the image name or not here
            
            # Uncomment the line below if you want to stop capturing after detecting a person
            break
        #return img_name, frame

    cam.release()
    cv2.destroyAllWindows()
    return img_name, frame

# Function to connect to SQLite database
def connect_db():
    return sqlite3.connect('names.db')

# Function to create the 'names' table if it doesn't exist
def create_table():
    conn = connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS names (id INTEGER PRIMARY KEY, name TEXT, image BLOB)''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_name', methods=['POST'])
def add_name():
    if request.method == 'POST':
        name = request.form['name']
        # Call capture_image() to capture an image
        img_name, captured_frame = capture_image_on_person()

        print("captured_frame",captured_frame.shape)
        print("img_name",img_name)
        label_Y = 2
        # Convert captured frame to JPEG format
        _, buffer = cv2.imencode('.jpg', captured_frame)
        jpeg_data = buffer.tobytes()

        # Store the name and image in the database
        conn = connect_db()
        c = conn.cursor()
        c.execute('INSERT INTO names (name, image) VALUES (?, ?)', (name, sqlite3.Binary(jpeg_data)))
        conn.commit()
        conn.close()

        (h, w) = captured_frame.shape[:2]  # Extracting just height and width
        # Resizing image to (300,300), performing no scaling (since scalefactor = 1),
        # size of blob/output is (300,300), mean subtraction of channels (BGR)
        blob = cv2.dnn.blobFromImage(cv2.resize(captured_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        cvNet.setInput(blob)
        detections = cvNet.forward() 
        for i in range(0, detections.shape[2]): # in this case it went through 200 iterations , since 200 detections were detected (200 bounding boxes)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #3:7 will give bounding box details
            (startX, startY, endX, endY) = box.astype("int")
            frame = captured_frame[startY:endY, startX:endX] #defining the region of interest (ROI)
            confidence = detections[0, 0, i, 2] #confidence is stored in 2
            print("*********************confidence*********",confidence)
            if confidence > 0.2:
                im = cv2.resize(frame,(124,124)) #why resize ? because we are giving this image to NN and we defined an input size of 124
                im = np.array(im)/255.0
                im = im.reshape(1,124,124,3)
                result = model.predict(im)
                print
                if result>0.5:
                    label_Y = 1
                else:
                    label_Y = 0

    if (label_Y==2):
        p = "Image is not captured properly "
        return render_template('output.html', submitted_name=p)

    elif (label_Y == 1):
        p = "You have not worn a mask. Please wear a mask and come back !! "
        return render_template('output.html', submitted_name=p)
    else:
        p = "Congratulations for obeying the guidelines. You may now proceed for the vacination !"
        return render_template('output.html', submitted_name=p)
      
                    
if __name__ == '__main__':
    create_table()
    app.run(debug=True)
