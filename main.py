import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from fastapi import FastAPI
import pickle 
app=FastAPI()

# Set paths
image_dir ='C:\My_Vs_Code_Projects\MyPythonProjects\AIMLproject\DeepLearningProjects\cnn_image_classification\image'
img_size = (100, 100)
batch_size = 32

# Get all image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
# print(image_paths)
random.shuffle(image_paths)

# Load images and labels
X = []
y = []

for image_path in image_paths:
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        print(f"img 1{img}")
        img = cv2.resize(img, img_size)
        print(f"img 2{img}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"img 3{img}")
        X.append(img)
        
        # Get label from filename==
        filename = os.path.basename(image_path)
       
        if filename.startswith('cat'):
            y.append(0)  # 0 for cat
        elif filename.startswith('dog'):
            y.append(1)  # 1 for dog
        else:
            print(f"Unknown file: {filename}, skipping")
            X.pop()  # Remove the appended image if label is invalid
            continue
            
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue

# Convert to numpy arrays and normalize

X = np.array(X, dtype="float32") / 255.0


#[241/255, 245/255, 246/255] ≈ [0.945, 0.961, 0.965]
y = np.array(y)

print(f"Loaded {len(X)} images with shape {X.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=batch_size,
    verbose=1
)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
pickle.dump(model,open("DLimgmodel.pkl",'wb'))
