# face_recognition_opencv
face-recognition project
Face Recognition using OpenCV
This repository contains a Python-based face recognition system built with OpenCV. It can detect and recognize faces in real-time using your webcam or process images and videos. It's simple, fast, and easy to integrate into other projects.
🧠 Features
•	• Real-time face detection and recognition
•	• Uses Haar Cascades or DNN for face detection
•	• Recognizes known faces using LBPHFaceRecognizer
•	• Easy to train with new faces
•	• Save and load trained models
•	• Capture images directly from webcam
•	• Modular and easy-to-understand code
📁 Project Structure

face_recognition_opencv/
│
├── dataset/               # Folder to store training images
├── trainer/               # Folder to store trained model (YAML)
├── haarcascade/           # Contains Haar Cascade XML files
├── captured_faces/        # Saved images during training
├── train_model.py         # Script to train the recognizer
├── face_recognition.py    # Main script for face detection and recognition
├── capture_faces.py       # Capture and label images for training
├── requirements.txt       # List of dependencies
└── README.md              # This file

🚀 Installation
1. Clone the repository:
git clone https://github.com/your-username/face_recognition_opencv.git
cd face_recognition_opencv
2. Install the required packages:
pip install -r requirements.txt
3. Make sure OpenCV is installed:
pip install opencv-python opencv-contrib-python
🏁 Usage
1. Capture face data:
Run the script to capture images for a new person:
python capture_faces.py
Follow the prompt to enter a person's ID and name.
2. Train the model:
After collecting images, train the face recognizer:
python train_model.py
3. Run face recognition:
Once trained, run the real-time face recognition:
python face_recognition.py
🧪 Requirements
•	• Python 3.6+
•	• OpenCV (opencv-python, opencv-contrib-python)
•	• NumPy
Install them with:
pip install opencv-python opencv-contrib-python numpy
🔧 Customization
• To use DNN-based face detection instead of Haar Cascades, modify the detection section in face_recognition.py.
• Update the dataset/ and trainer/ paths as needed for your own organization.
🛡️ License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
🔗 Connect
• GitHub: https://github.com/satheprasad
• Email: Satheprasad50@gmail.com
