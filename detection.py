import cv2
import numpy as np
import os

# Path to the directory containing face images for training
data_path = 'C:/Users/SAI/PycharmProjects/pythonProject/dataset'

# Get a list of all image files in the directory
onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

# Initialize lists to store training data and corresponding labels
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = os.path.join(data_path, onlyfiles[i])
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

# Create and train the LBPH face recognizer
model = cv2.face_LBPHFaceRecognizer.create()

if len(Training_Data) > 0:
    model.train(Training_Data, Labels)
    model.save('trained_model.yml')
else:
    print("Dataset Model Training Complete!!!!!!")

# Load the Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier('C:/Users/SAI/PycharmProjects/pythonProject/haarcascade_frontalface_default.xml')

# Load an image for face recognition
image_path = 'C:\\Users\\SAI\\PycharmProjects\\pythonProject\\image\\WhatsApp Image 2023-09-01 at 14.24.31.jpg'
image = cv2.imread(image_path)

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1 - (result[1])/300))
            cv2.putText(image, f"Confidence: {confidence}%", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if confidence > 80:
                cv2.putText(image, "Prasad Sathe", (250, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Recognition', image)
            else:
                cv2.putText(image, "Unknown", (250, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', image)
        else:
            cv2.putText(image, "Unknown", (250, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)

    except:
        cv2.putText(image, "Face Not Found", (250, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', image)
        pass

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
