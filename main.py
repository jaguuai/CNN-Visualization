# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:56:58 2024

@author: CNN-Visualization

"""
# load vgg modela
from keras.applications.vgg19 import VGG19
# load the model
model=VGG19()
# summarize the model
model.summary()
# show the layer numbers
# n=0
# # for layer in model.layers:
# #     print(n,layer.name)
# #     n+=1
# # shows the filter shape and bias shape
# for layer in model.layers:
#     if "conv" in layer.name:
#         filters,biases=layer.get_weights()
#         print(n,layer.name,filters.shape,biases.shape)
#     n+=1
from matplotlib import pyplot as plt
# Retrieve weights from the first convolutional layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels:", s[2])    
print("Filter size:", s[0], s[1]) 
print("Total number of filters:", s[3])

# Normalize filter values to 0-1 for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot first few filters
n_filters, ix = 4, 1
fig = plt.figure(figsize=(12, 12))

for i in range(n_filters):
    # Get filter
    f = filters[:, :, :, i]
    
    # Plot each channel separately
    for j in range(s[2]):  # s[2] is the number of color channels
        ax = plt.subplot(n_filters, s[2], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot the 2D slice
        plt.imshow(f[:, :, j], cmap="gray")
        ix += 1
        
# Show the figure
plt.show()


def plot_feature_maps(feature_maps):
    # plot all feature maps
    col=8
    row=int(feature_maps.shape[3]/col)
    ix=1
    plt.figure(figsize=(20,20))
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax=plt.subplot(row,col,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0,:,:,ix-1],cmap="gray")
            ix+=1
    plt.show()
    
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# load the model
model=VGG19 ()

#Select the hidde layer to visualize
n=1

# redefine model to output right after the hidden layer
model = Model (inputs=model.inputs, outputs=model.layers [n].output)
model.summary()

# load the image with the required shape
img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img=img_to_array(img)
# expand dimensions so that it represents a single 'sample
img = expand_dims(img, axis=0)
# prepare the image (e.q. scale pixel values for the vgg)
img = preprocess_input (img)
# get feature map for first hidden layer
feature_maps=model.predict (img)
print("Feature maps:",feature_maps.shape)

#plot all the feature maps
plot_feature_maps(feature_maps)












