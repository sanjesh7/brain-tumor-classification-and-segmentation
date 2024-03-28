import numpy as np 
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
import os
from flask import Flask, render_template, request,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, disk
from collections import deque


def tumorSegmentation(img_path):
    image=Image.open(img_path)


# Increase contrast using PIL
    contrast_factor = 1.32  
    enhancer = ImageEnhance.Contrast(image)
    image_contrasted = enhancer.enhance(contrast_factor)

# Convert PIL Image to OpenCV format
    image_contrasted_cv = cv2.cvtColor(np.array(image_contrasted), cv2.COLOR_RGB2BGR)

# Convert the contrast-enhanced image to grayscale using OpenCV
    original_image = cv2.cvtColor(image_contrasted_cv, cv2.COLOR_BGR2GRAY)

# Resize the original image to 224x224
    resized_image = cv2.resize(original_image, (224, 224))
    sum=0
    k=0
    for i in range(resized_image.shape[0]):
        for j in range(resized_image.shape[1]):
            if resized_image[i][j]>20:
                sum=sum+resized_image[i][j]
                k=k+1

    t_value=(sum/k)+30

    print(t_value)
# Convert the resized image to binary
    _, binary_image = cv2.threshold(resized_image, t_value, 255, cv2.THRESH_BINARY)
    

# Convert the binary image to a numpy array
    binary_array = np.array(binary_image)

# Apply the horizontal operation
    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1]):
            if j < 100:  # Checking condition similar to j<68
                if binary_array[i, j] == 255:
                    for k in range(j, min(binary_array.shape[1], j + 20)):
                        binary_array[i, k] = 0
                    break

    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1] - 1, -1, -1):
            if j > 101:  # Checking condition similar to j>224-67
                if binary_array[i, j] == 255:
                    for k in range(j, max(0, j - 20), -1):
                        binary_array[i, k] = 0
                    break

# vertical operation
    for i in range(30):
        for j in range(68, 157):
            if binary_array[i, j] == 255:
                for k in range(i, min(binary_array.shape[0], i + 15)):
                    binary_array[k, j] = 0

    for i in range(binary_array.shape[0] - 1, binary_array.shape[0] - 60, -1):
        for j in range(68, 157):
            if binary_array[i, j] == 255:
                for k in range(i, max(0, i - 20), -1):
                    binary_array[k, j] = 0

    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1]):
            if i<56:
                binary_array[i][j]=0

    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1]):
            if i>180:
                binary_array[i][j]=0
            
    for i in range(0,60):
        for j in range(0,224):
            binary_array[i][j]=0



# Apply erosion and dilation using skimage
    eroded_image = binary_erosion(binary_array, disk(3.5))

    dilated_image = binary_dilation(eroded_image, disk(2.5))

# Check if the dilated image contains some white cells or not 
    sum = 0
    for i in range(dilated_image.shape[0]):
        for j in range(dilated_image.shape[1]):
            if dilated_image[i, j] == 1:
                sum += 1
    if sum>15:
        ans="yes"
    else:
        ans="no"

    if ans=="no":
        return binary_array




    def bfs(image, visited, start_x, start_y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        cluster_size = 0
        cluster_indices = []
    
        queue = deque([(start_x, start_y)])
        visited[start_x][start_y] = True
    
        while queue:
            x, y = queue.popleft()
            cluster_size += 1
            cluster_indices.append((x, y))
        
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and image[nx][ny] == 1 and not visited[nx][ny]:
                    queue.append((nx, ny))
                    visited[nx][ny] = True
    
        return cluster_size, cluster_indices

    def find_largest_cluster(image):
        visited = np.zeros_like(image, dtype=bool)
        largest_cluster_size = 0
        largest_cluster_indices = []
    
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == 1 and not visited[i][j]:
                    cluster_size, cluster_indices = bfs(image, visited, i, j)
                    if cluster_size > largest_cluster_size:
                        largest_cluster_size = cluster_size
                        largest_cluster_indices = cluster_indices
    
    # Create a new image with only the pixels from the largest cluster set to 1
        new_image = np.zeros_like(image)
        for idx in largest_cluster_indices:
            new_image[idx[0]][idx[1]] = 1
    
        return largest_cluster_size, largest_cluster_indices, new_image


    
    largest_cluster_size, largest_cluster_indices, updated_image = find_largest_cluster(dilated_image)




    img=binary_image


    def connected_pixels(image, start_index):
        height, width = len(image), len(image[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    # Function to check if a pixel is within the bounds of the image
        def is_valid(x, y):
            return 0 <= x < height and 0 <= y < width

    # Initialize visited array and queue for BFS
        visited = [[False] * width for _ in range(height)]
        queue = deque([start_index])
        visited[start_index[0]][start_index[1]] = True

        connected_pixels_indices = []

        while queue:
            x, y = queue.popleft()
            connected_pixels_indices.append((x, y))

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if is_valid(new_x, new_y) and not visited[new_x][new_y] and image[new_x][new_y] == 255:
                    queue.append((new_x, new_y))
                    visited[new_x][new_y] = True

        return connected_pixels_indices



    size = len(largest_cluster_indices) // 2
    idx = largest_cluster_indices[size]
    values=connected_pixels(binary_image, idx)


    def visualize_connected_pixels(image, connected_pixels_indices):
    # Create a copy of the image to modify
        modified_image = np.array(image)

    # Set pixels at connected indices to 255 and others to 0
        for i in range(len(image)):
            for j in range(len(image[0])):
                if (i, j) in connected_pixels_indices:
                    modified_image[i][j] = 255
                else:
                    modified_image[i][j] = 0
        return modified_image

    
    final_image=visualize_connected_pixels(img, values)
    return final_image






app = Flask(__name__)
model = load_model("./bwd2.keras")
class_names = ['glioma Tumor', 'meningioma Tumor', 'No Tumor', 'pituitary Tumor']

def predict_image_class(model, class_names, img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    return predicted_class_name, predicted_probability



@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classifier')
def classification():
    return render_template('classifier.html')

@app.route('/classifier', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        file = request.files['image']
        img_path = 'static/uploads/temp_image.jpg'
        print(type(file))
        file.save(img_path)
        # Predict image class
        predicted_class, predicted_prob = predict_image_class(model, class_names, img_path)
        if predicted_class!='No Tumor':
            tumor_image=tumorSegmentation(img_path)
            save_path="static/uploads/tumor-image.jpg"
            cv2.imwrite(save_path, tumor_image)
            return render_template('classifier.html', predicted_class=predicted_class, predicted_prob=predicted_prob, img_path=img_path,img=save_path)

            
        return render_template('classifier.html', predicted_class=predicted_class, predicted_prob=predicted_prob, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)