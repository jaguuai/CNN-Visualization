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