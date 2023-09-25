import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

classifier = load_model('MNIST_data_model.keras')
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_X = test_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0

num_classes = 10
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)

# Build a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='SAME'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

batch_size = 32
epochs = 10  
history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_y))

drawing = False
black_image = np.zeros((256, 256, 3), np.uint8)
cv2.namedWindow('Input Here')

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

def draw_circles(event, x, y, flags, param):
    global drawing, black_image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(black_image, (x, y), 5, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.setMouseCallback('Input Here', draw_circles)

while True:
    cv2.imshow('Input Here', black_image)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  
        break
    elif key == ord('c'):  # Press 'c' to clear the canvas
        black_image = np.zeros((256, 256, 3), np.uint8)
    elif key == ord('p'):  # Press 'p' to predict the digit
        input_img = preprocess_image(black_image)
        prediction = model.predict(input_img)
        digit = np.argmax(prediction)
        print(f"Predicted Digit",digit)
