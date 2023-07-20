# importing the python open cv library
import cv2

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from glob import glob

from tensorflow.keras import backend as K
from tensorflow.keras.applications import mobilenet, inception_v3, inception_resnet_v2, resnet50
from tensorflow.keras.layers import Dense, dot, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


'''*************************************Parameters*******************************************************************'''

# data import options
# class_list_filename = "siamese_testSet_classes.txt"
root_dir = "data/product"
product_name = []
# siamese model parameters
img_shape = (224, 224, 3)
feature_shape = (2048,)
base_id = 1  # 0 = Inception-v3, 1 = MobileNet, 2 = InceptionResNet-v2, 3 = ResNet50

# saved training weights to be loaded
weights_filename = 'product_mobileNet.h5'
# test loop
acc_array = []

'''******************************************************************************************************************'''

'''*************************************Functions********************************************************************'''



# create a base model that generates a (n,1) feature vector for an input image
def create_base_model(input_shape):
    base = mobilenet.MobileNet(input_shape=input_shape, weights='imagenet', include_top=False)
    base_name = 'MobileNet'
    print("\nBase Network: %s" % base_name)
    top = GlobalAveragePooling2D()(base.output)
    # freeze all layers in the base network
    for layers in base.layers:
        layers.trainable = False
    model = Model(inputs=base.input, outputs=top, name='base_model')
    return model


# calculate cosine distance b/t feature vector outputs from base network
def cos_distance(feat_vects):
    x1, x2 = feat_vects
    result = dot(inputs=[x1, x2], axes=1, normalize=True)
    return result


# create a siamese model
def create_siamese(base, input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    encoding_a = base(input_a)
    encoding_b = base(input_b)
    fc1_a = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc1_a')(encoding_a)
    fc1_b = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc1_b')(encoding_b)
    distance = Lambda(function=cos_distance, name='cos_distance', )([fc1_a, fc1_b])
    prediction = Dense(units=1, activation='sigmoid', name='sigmoid')(distance)
    model = Model(inputs=[input_a, input_b], outputs=prediction, name='siamese_model')
    # load weights for sigmoid layers
    path_to_weights = "weights/" + weights_filename
    model.load_weights(filepath=path_to_weights, by_name=True)
    print('Loading weights from ' + path_to_weights)
    return model

'''*************************************Main*************************************************************************'''
if __name__ == ("__main__"):

    K.clear_session()
    # create base model
    base_network = create_base_model(input_shape=img_shape)

    # create siamese network
    siamese_model = create_siamese(base=base_network, input_shape=img_shape)

    # test model with binary_cross-entropy loss function
    siamese_model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    # test model
    def preprocess(img_path):
        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, (224, 224), img_cv)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = np.array(img_cv) / 255.0
        new_img=np.expand_dims(img_cv,axis=0)
        new_img = np.array(new_img, dtype=np.float32)
        return new_img

    def classify(img_path,model):
        accuracy_max = 0
        train_path = 'data/product/train'
        label = 'unidentified'
        img = preprocess(img_path)
        for classname in glob(train_path+ '/*'):
            for sample_path in glob(classname+'/*'):
                sample = preprocess(sample_path)
                predict =  model.predict(x=[img, sample], batch_size=1, verbose=0)
                accuracy = predict[0]
                if accuracy > accuracy_max:
                    accuracy_max = accuracy
                    label = classname.split('/')[-1]
                break
        return accuracy_max, label

    img_path = 'data/product/train/TEA1/TEA1_0002.png'
    acc, label = classify(img_path, siamese_model)
    print(acc, label)

# intialize the webcam and pass a constant which is 0
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# title of the app
cv2.namedWindow('python webcam screenshot app')

# let's assume the number of images gotten is 0
img_counter = 0

# while loop
while True:
    # intializing the frame, ret
    ret, frame = cam.read()
    # if statement
    if not ret:
        print('failed to grab frame')
        break
    # the frame will show with the title of test
    cv2.imshow('test', frame)
    #to get continuous live video feed from my laptops webcam
    k  = cv2.waitKey(1)
    # if the escape key is been pressed, the app will stop
    if k%256 == 27:
        print('escape hit, closing the app')
        break
    # if the spacebar key is been pressed
    # screenshots will be taken
    elif k%256  == 32:
        # the format for storing the images scrreenshotted
        img_name = f'opencv_frame_{img_counter}.png'
        # saves the image as a png file
        cv2.imwrite(img_name, frame)
        print('screenshot taken')
        # the number of images automaticallly increases by 1
        img_counter += 1

# release the camera
cam.release()

# stops the camera window
# cam.destoryAllWindows()




img1 = preprocess('opencv_frame_0.png')
img2 = preprocess('opencv_frame_1.png')
predict =  siamese_model.predict(x=[img1, img2], batch_size=1, verbose=0)
print(predict)

fig,ax = plt.subplots(1,2,  figsize=(16,12))
ax[0].imshow(img1[0])
ax[1].imshow(img2[0])
plt.title(f'Similarity {predict[0][0]}')
plt.show()