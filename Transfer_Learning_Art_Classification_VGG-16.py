# Importing libraries, APIS & Tensorflow

# import future

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow & keras

import tensorflow as tf

# import modules, libraries, APIs, optimizers, layers, etc...


from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from scipy.interpolate import make_interp_spline
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import PIL
from PIL import Image
from contextlib import redirect_stdout
import sys


# import glob for directories

import glob

# print file name

file_name =  os.path.basename(sys.argv[0])
file_name = str(file_name)

# Trim the ".py" extensiojn from the file 

file_name = file_name[:-3]

print("Name of the File:", file_name)

# set another (very high) limit for image processing

PIL.Image.MAX_IMAGE_PIXELS = 9933120000

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# setting the paths, pathfiles

# setting the paths, pathfiles

parent_dir = glob.glob(r'C:\**\SeAC Dataset/', recursive=True)
parent_dir = str(parent_dir[0])

# as the glob library provides the path in "\\" format we need to change it to "/"

parent_dir = parent_dir.replace('\\', '/')

path_train_images = parent_dir

saved_model_path = glob.glob(r'C:\**\Saved_Models/', recursive=True)
saved_model_path = str(saved_model_path[0])
saved_model_path = saved_model_path.replace('\\', '/')

path_scripts = glob.glob(r'C:\**\Scripts/', recursive=True)
path_scripts = str(path_scripts[0])
path_scripts = path_scripts.replace('\\', '/')


### Setting up all the parameters for the training, processing of data and produced files management ###

# import the labels of the dataset

data = pd.read_excel(train_labels_path, engine="openpyxl")

# check the dataframe to inspect any NaNs or missing values

print(data.head(3))

# check the number of rows and columns of the dataframe

print(data.shape)

### setting parameters for the saving and loading of produced files ###

# set batch size

# num_batch = 10

# adjusting the size of images into the same dimensions

img_width = 250
img_height = 250

# adjsusting training parameters

lrn_rate = 0.0001  # learning rate
num_epochs = 10  # number of training epochs
act_func = "sigmoid"  # activation fuction

num_neurons_1st_dense = 70 # number of neurons in first dense layer
drop_perc_1st_dense = 0 # percentage of dropout rate in first dense layer

num_neurons_2nd_dense = 35 # number of neurons in second dense layer
drop_perc_2nd_dense = 0 # percentage of dropout rate in second dense layer

num_neurons_3d_dense = 0 # number of neurons in third dense layer
drop_perc_3d_dense = 0 # percentage of dropout rate in third dense layer

# create an empty list to take in the images

X = []

# attach the images with the labels, in order to have supervised learning and convert them to have values between 0 - 1 by multiplying with 255 (RGB values: 0 - 255)

for i in tqdm(range(data.shape[0])):
    path = (
        path_train_images
        + data["Art_Genre"][i]
        + "/"
        + data["Artist_Name"][i]
        + "/"
        + data["Painting_Name"][i]
        + ".jpg"
    )
    img = image.load_img(path, target_size=(img_width, img_height, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    X.append(img)

# convert images into np array

X = np.array(X)

# get the size of dataframe X

print("The shape of X list is: ", X.shape)

# removing "unnecessary columns, in order to have only the one-hot encoded columns for the training

y = data.drop(["Artist_Name", "Painting_Name", "Art_Genre"], axis=1)
y = np.array(y)

# inspect the size of new labels dataframe and make sure that columns match the number of classes

print("The shape of y list is: ", y.shape)

# set that the evaluation set is 10% of the total dataset
# the testing dataset will be 20% of the total dataset
# the final percentage of training set will be 70% of the total dataset


# create the train/test sets, set the size of test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create the test/validation sets, set the size of validation set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

# get the shapes of train, test and validation splits

print("The shape of X_train is: ", X_train.shape)
print("The shape of y_train is: ", y_train.shape)

print("The shape of X_test is: ", X_test.shape)
print("The shape of y_test is: ", y_test.shape)

print("The shape of X_val is: ", X_val.shape)
print("The shape of y_val is: ", y_val.shape)



# create the convolutional network

# set the model as sequential
# imply batchnormalization in every layer to make training faster
# imply dropout in each hidden layer to avoid overfitting


# Load the VGG model

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

"""" As this is a database of a fair size, we are choosing to unfreeze all the layers, to get the optimal result.

     In databases of smaller size, it will may be needed to freeze more layers, if not all""""


for layer in vgg_conv.layers[:]:
   layer.trainable = False
   
for layer in vgg_conv.layers[:]:
   layer.trainable = True
    
# Check the trainable status of the individual layers

for layer in vgg_conv.layers:
    print(layer.name, layer.trainable)
    
# Create the model

model = models.Sequential()

# Add the vgg convolutional base model

model.add(vgg_conv)

# flatten all layers

model.add(layers.Flatten())

# Add new layers

# Add first additional dense layer

model.add(layers.Dense(num_neurons_1st_dense, activation='relu'))
#model.add(layers.Dropout(drop_perc_1st_dense))

# Add second additional dense layer

model.add(layers.Dense(num_neurons_2nd_dense, activation='relu'))
#model.add(layers.Dropout(drop_perc_2nd_dense))

# Add third additional dense layer (if needed)

#model.add(layers.Dense(num_neurons_3d_dense, activation='relu'))
#model.add(layers.Dropout(drop_perc_3d_dense))

model.add(layers.Dense(10, activation= act_func))

# Show a summary of the model. Check the number of trainable parameters

model.summary()    
    
# Compile the model

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lrn_rate),
              metrics=['acc'])


history = model.fit(X_train, y_train,
              batch_size=64,
              epochs=num_epochs,
              validation_data=(X_test, y_test))
              

print(history.history.keys())


# plot the accuracy for train/test sets to check for overfitting or underfitting (less possible)
# plot the loss for train/test sets for same reasons

# plot the accuracy for train/test sets to check for overfitting or underfitting (less possible)
# plot the loss for train/test sets for same reasons

# Create smoother lines in the graphs

epoch_list = []

acc_list = history.history["acc"]
print('accuracy list:', acc_list)

val_acc_list = history.history["val_acc"]
print('validation accuracy list:', val_acc_list)

loss_list = history.history["loss"]
print('loss list:', loss_list)

val_loss_list = history.history["val_loss"]
print('validation loss list:', val_loss_list)

epoch_lim = num_epochs + 1

for i in range(1, epoch_lim):
    epoch_list.append(i)
    
x_plot = np.array(epoch_list)

y_plot_acc = np.array(history.history["acc"])
y_plot_val_acc = np.array(history.history["val_acc"])

X_Y_Spline_acc = make_interp_spline(x_plot, y_plot_acc)
X_Y_Spline_val_acc = make_interp_spline(x_plot, y_plot_val_acc)


# Returns evenly spaced numbers
# over a specified interval.

X_plot = np.linspace(x_plot.min(), x_plot.max())
y_plot_acc = X_Y_Spline_acc(X_plot)
y_plot_val_acc = X_Y_Spline_val_acc(X_plot)

# Plotting the Graph

plt.plot(X_plot, y_plot_acc)
plt.plot(X_plot, y_plot_val_acc)
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="lower right")
plt.savefig(
    saved_model_path
    + file_name + "_model_accuracy.png"
)

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

y_plot_loss = np.array(history.history["loss"])
y_plot_val_loss = np.array(history.history["val_loss"])

X_Y_Spline_loss = make_interp_spline(x_plot, y_plot_loss)
X_Y_Spline_val_loss = make_interp_spline(x_plot, y_plot_val_loss)

# Returns evenly spaced numbers
# over a specified interval.

y_plot_loss = X_Y_Spline_loss(X_plot)
y_plot_val_loss = X_Y_Spline_val_loss(X_plot)

# Plotting the Graph

plt.plot(X_plot, y_plot_loss)
plt.plot(X_plot, y_plot_val_loss)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.savefig(
    saved_model_path
    + file_name + "_model_loss.png"
)

print("Train procedure was completed successfully")

# testing of model's predicting ability in unknown sample

print("Testing procedure started")


# make predictions

predictions = model.predict(X_val)

# get the max values of the predictions and true values to get the metric values

predictions = np.argmax(predictions, axis=1)
y_val = np.argmax(y_val, axis=1)

accuracy_rate = accuracy_score(y_val, predictions)

accuracy_rate = Accuracy_rate * 100
accuracy_rate = round(Accuracy_rate, 2)

print("Success rate: {} %".format(pososto_epityxias))


# create a txt file to save the training parameters of the created model

with open(
    (
        saved_model_path
        + file_name + "_configuration.txt"
    ),
    "w",
) as f:
    f.write(
        "Successful identification rate: {} % \nNumber of epochs: {} \nLearning rate: {} \nImage height and width: {} \nActivation function: {} \naccuracy list: {} \nvalidation accuracy list: {} \nloss list: {} \nvalidation loss list: {} \n\n\n\n\n\n\n\n{}".format(
            accuracy_rate,
            num_epochs,
            lrn_rate,
            img_width,
            act_func,
            acc_list,
            val_acc_list,
            loss_list,
            val_loss_list,
            model.summary(),
        )
    )

# add the model summary to the created txt file

with open(
    (
        saved_model_path
        + file_name + "_configuration.txt"
    ),
    "a",
) as f:
    with redirect_stdout(f):
        model.summary()
        
# construct a confussion matrix to examine in which style the code performed better

conf_mat = confusion_matrix(y_val, predictions)

# get the names of the classes that correspond to the binary data-predictions/true values

cols_list = data['Art_Genre'].unique()
class_names = cols_list

# make a dataframe of the confusion matrix

conf_mat_dataframe = pd.DataFrame(
    conf_mat, index=[i for i in class_names], columns=[i for i in class_names]
)

# plot and save the confusion matrix

plt.figure(figsize=(20, 16))
plt.rcParams.update({"font.size": 18})
plt.rcParams.update({"font.family": "georgia"})
plt.locator_params(axis="both", integer=True, tight=True)
plt.title("Confusion Matrix")
sn.heatmap(
    conf_mat_dataframe,
    annot=True,
    linewidths=0.1,
    linecolor="g",
    cbar=True,
    cmap="cividis",
)

plt.savefig(
    saved_model_path
    + file_name + "_confusion_matrix.png"
)


# open and set the config file to enter all the training parameters
# and performance results

df_config = pd.read_excel(saved_model_path + "Config.xlsx", engine = "openpyxl")

# set the number of row to enter the inputs from this script

input_placeholder = len(df_config)
input_placeholder = input_placeholder + 1

# enter the name of file in config dataset

df_config.at[input_placeholder, 'File'] = file_name

# enter the accuracy percentage in config dataset

df_config.at[input_placeholder, 'Rate (%)'] = accuracy_rate

# enter the dimensions of the input images in config dataset

df_config.at[input_placeholder, 'Hight/Width'] = img_width

# enter the epochs number in config dataset

df_config.at[input_placeholder, 'Epochs'] = num_epochs

# enter the learning rate value in config dataset

df_config.at[input_placeholder, 'Learning_Rate'] = lrn_rate

# enter the activation function in config dataset

df_config.at[input_placeholder, 'Act_Func.'] = act_func

# enter the number of neurons of first dense layer in config dataset

df_config.at[input_placeholder, 'Neuron_1st_Dense'] = num_neurons_1st_dense

# enter the dropout rate of first dense layer in config dataset

df_config.at[input_placeholder, 'Dropout_1st_Dense'] = drop_perc_1st_dense

# enter the number of neurons of second dense layer in config dataset

df_config.at[input_placeholder, 'Neuron_2nd_Dense'] = num_neurons_2nd_dense

# enter the dropout rate of second dense layer in config dataset

df_config.at[input_placeholder, 'Dropout_2nd_Dense'] = drop_perc_2nd_dense

# enter the number of neurons of third dense layer in config dataset

df_config.at[input_placeholder, 'Neuron_3d_Dense'] = num_neurons_3d_dense

# enter the dropout rate of third dense layer in config dataset

df_config.at[input_placeholder, 'Dropout_3d_Dense'] = drop_perc_3d_dense

# save the config dataset

df_config.to_excel(saved_model_path + "Config.xlsx",
                   engine='openpyxl',
                   index = False)
                   
### Rename the script file to add the accuracy percentage in the file name

file_before = path_scripts + file_name + ".py" 
file_after = path_scripts + file_name + "_(" + str(accuracy_rate) + ")"  + ".py"

os.rename(file_before, file_after)
                   
print("The script was executed successfully")

