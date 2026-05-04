import pandas as pd

# Path to the dataset legend file
csv_path = "facial_expressions-master/data/legend.csv"

# Load the CSV file
df = pd.read_csv(csv_path)

# Print all emotion classes and their counts
print("===== All emotion classes in the dataset =====\n")

class_counts = df["emotion"].value_counts()

for emotion, count in class_counts.items():
    print(f"{emotion} : {count}")

# Print total number of unique classes
print("\nTotal unique classes:", df["emotion"].nunique())
import pandas as pd
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Path to CSV file containing labels
legend_path = "facial_expressions-master/data/legend.csv"

# Path to image folder
img_dir = "facial_expressions-master/images"

# Load legend file
legend = pd.read_csv(legend_path)
# Convert all emotion labels to lowercase
legend["emotion"] = legend["emotion"].str.lower()

# Keep only happy and neutral related emotions
legend = legend[legend["emotion"].isin(["neutral", "happy", "happiness"])]

# Replace "happiness" with "happy" to unify labels
legend["emotion"] = legend["emotion"].replace({"happiness": "happy"})
# Define emotion classes
emotion_list = ["happy", "neutral"]

# Map emotions to numerical IDs
emotion_to_id = {emo: i for i, emo in enumerate(emotion_list)}
X = []
y = []

for _, row in legend.iterrows():

    img_path = os.path.join(img_dir, row["image"])

    # Skip if image does not exist
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)

    # Skip if image cannot be read
    if img is None:
        continue

    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image to 48x48
    img = cv2.resize(img, (48, 48))

    # Normalize pixel values
    img = img / 255.0

    X.append(img)
    y.append(emotion_to_id[row["emotion"]])
# Convert to numpy arrays
X = np.array(X).reshape(-1, 48, 48, 1)
y = to_categorical(np.array(y))
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential()

# First convolution layer
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

# Second convolution layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# Flatten feature maps
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Dropout to reduce overfitting
model.add(Dropout(0.3))

# Output layer (2 classes)
model.add(Dense(2, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=16,
    validation_data=(X_test, y_test)
)
model.save("emotion_model.keras")

print("Model saved successfully")
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained emotion model
model = load_model("emotion_model.keras")

# Define emotion class names
emotion_labels = ["happy", "neutral"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Crop the detected face region
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.reshape(1, 48, 48, 1) / 255.0
        
        # Predict emotion
        pred = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(pred)]
        
        # Draw rectangle & label around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, (0,255,0), 2
        )
    
    # Display live results
    cv2.imshow("Emotion Detection", frame)
    
    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
