

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras as keras
import tensorflow as tf
import os
import numpy as np
import cv2

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
#print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)



def create_model():
  input = keras.layers.Input(shape=(224,224,3))

  x = keras.layers.Conv2D(64, (7,7),strides=(2, 2),padding='same',use_bias=False)(input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)

  #identity_block1-1
  x1_1 = keras.layers.Conv2D(64, (3,3), strides=(1, 1),padding='same',use_bias=False)(x)
  x1_1 = keras.layers.BatchNormalization()(x1_1)
  x1_1 = keras.layers.Activation('relu')(x1_1)

  x1_1 = keras.layers.Conv2D(64, (3,3), strides=(1, 1),padding='same',use_bias=False)(x1_1)
  x1_1 = keras.layers.BatchNormalization()(x1_1)


  added1_1 = keras.layers.Add()([x1_1, x])
  added1_1 = keras.layers.Activation('relu')(added1_1)

  #identity_block1-2
  x1_2 = keras.layers.Conv2D(64, (3,3), strides=(1, 1),padding='same',use_bias=False)(added1_1)
  x1_2 = keras.layers.BatchNormalization()(x1_2)
  x1_2 = keras.layers.Activation('relu')(x1_2)

  x1_2 = keras.layers.Conv2D(64, (3,3), strides=(1, 1),padding='same',use_bias=False)(x1_2)
  x1_2 = keras.layers.BatchNormalization()(x1_2)


  added1_2 = keras.layers.Add()([x1_2, added1_1])
  added1_2 = keras.layers.Activation('relu')(added1_2)


  #conv_bloack2
  x2 = keras.layers.Conv2D(128, (3,3), strides=(2, 2),padding='same',use_bias=False)(added1_2)
  x2 = keras.layers.BatchNormalization()(x2)
  x2 = keras.layers.Activation('relu')(x2)

  x2 = keras.layers.Conv2D(128, (3,3), strides=(1, 1),padding='same',use_bias=False)(x2)
  x2 = keras.layers.BatchNormalization()(x2)

  short_cut2 = keras.layers.Conv2D(128, (1,1), strides=(2, 2),padding='valid',use_bias=False)(added1_2)
  short_cut2 = keras.layers.BatchNormalization()(short_cut2)

  added2 = keras.layers.Add()([x2, short_cut2])
  added2 = keras.layers.Activation('relu')(added2)

  #identity_block2-1
  x2_1 = keras.layers.Conv2D(128, (3,3), strides=(1, 1),padding='same',use_bias=False)(added2)
  x2_1 = keras.layers.BatchNormalization()(x2_1)
  x2_1 = keras.layers.Activation('relu')(x2_1)

  x2_1 = keras.layers.Conv2D(128, (3,3), strides=(1, 1),padding='same',use_bias=False)(x2_1)
  x2_1 = keras.layers.BatchNormalization()(x2_1)

  added2_1 = keras.layers.Add()([x2_1, added2])
  added2_1 = keras.layers.Activation('relu')(added2_1)

  #conv_bloack3
  x3 = keras.layers.Conv2D(256, (3,3), strides=(2, 2),padding='same',use_bias=False)(added2_1)
  x3 = keras.layers.BatchNormalization()(x3)
  x3 = keras.layers.Activation('relu')(x3)

  x3 = keras.layers.Conv2D(256, (3,3), strides=(1, 1),padding='same',use_bias=False)(x3)
  x3 = keras.layers.BatchNormalization()(x3)
  x3 = keras.layers.Activation('relu')(x3)

  short_cut3 = keras.layers.Conv2D(256, (1,1), strides=(2, 2),padding='valid',use_bias=False)(added2_1)
  short_cut3 = keras.layers.BatchNormalization()(short_cut3)

  added3 = keras.layers.Add()([x3, short_cut3])
  added3 = keras.layers.Activation('relu')(added3)


  #identity_block3-1
  x3_1 = keras.layers.Conv2D(256, (3,3), strides=(1, 1),padding='same',use_bias=False)(added3)
  x3_1 = keras.layers.BatchNormalization()(x3_1)
  x3_1 = keras.layers.Activation('relu')(x3_1)

  x3_1 = keras.layers.Conv2D(256, (3,3), strides=(1, 1),padding='same',use_bias=False)(x3_1)
  x3_1 = keras.layers.BatchNormalization()(x3_1)

  added3_1 = keras.layers.Add()([x3_1, added3])
  added3_1 = keras.layers.Activation('relu')(added3_1)

  #conv_bloack4
  x4 = keras.layers.Conv2D(512, (3,3), strides=(2, 2),padding='same',use_bias=False)(added3_1)
  x4 = keras.layers.BatchNormalization()(x4)
  x4 = keras.layers.Activation('relu')(x4)

  x4 = keras.layers.Conv2D(512, (3,3), strides=(1, 1),padding='same',use_bias=False)(x4)
  x4 = keras.layers.BatchNormalization()(x4)

  short_cut4 = keras.layers.Conv2D(512, (1,1), strides=(2, 2),padding='valid',use_bias=False)(added3_1)
  short_cut4 = keras.layers.BatchNormalization()(short_cut4)

  added4 = keras.layers.Add()([x4, short_cut4])
  added4 = keras.layers.Activation('relu')(added4)


  #identity_block4-1
  x4_1 = keras.layers.Conv2D(512, (3,3), strides=(1, 1),padding='same',use_bias=False)(added4)
  x4_1 = keras.layers.BatchNormalization()(x4_1)
  x4_1 = keras.layers.Activation('relu')(x4_1)

  x4_1 = keras.layers.Conv2D(512, (3,3), strides=(1, 1),padding='same',use_bias=False)(x4_1)
  x4_1 = keras.layers.BatchNormalization()(x4_1)

  added4_1 = keras.layers.Add()([x4_1, added4])
  added4_1 = keras.layers.Activation('relu')(added4_1)



  x5 = keras.layers.AveragePooling2D((7,7))(added4_1)
  x5 = keras.layers.Flatten()(x5)
  x5 = keras.layers.Dense(2, activation='softmax')(x5)

  model = keras.models.Model(input, x5)
  return model



def create_training_data():

  img_training_cat_x = np.array([])
  img_training_dog_x = np.array([])

  for k in range(100):
    img =  cv2.imread('/content/resnet_keras/training_data/cat1/cat.' + str(k)+'.jpg')
    img =  cv2.resize(img,(224,224), interpolation =  cv2.INTER_AREA)
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    img = img.reshape((1,224,224,3))
    if k ==0:
      img_training_cat_x = img
    else:
      img_training_cat_x = np.vstack((img_training_cat_x, img))

  for k in range(100):
    img =  cv2.imread('/content/resnet_keras/training_data/dog1/dog.' + str(k)+'.jpg')
    img =  cv2.resize(img,(224,224), interpolation =  cv2.INTER_AREA)
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    img = img.reshape((1,224,224,3))
    if k ==0:
      img_training_dog_x = img
    else:
      img_training_dog_x = np.vstack((img_training_dog_x, img))

  training_data_x = np.vstack((img_training_cat_x,img_training_dog_x))

  a = np.zeros((100,2))
  b = np.zeros((100,2))
  for i in range(100):
    b[i][1] = 1
  training_data_y = np.vstack((a,b))

  return training_data_x, training_data_y

training_x,training_y =create_training_data()

#print(training_x.shape, training_y.shape)

with strategy.scope():
  model = create_model()
  #model.summary()
  model.compile(optimizer= RMSprop(lr = 0.001),
                experimental_steps_per_execution = 50,
                loss='categorical_crossentropy',
                metrics=['acc'])


history = model.fit(
    x = training_x, y = training_y, batch_size=64, epochs=100, verbose = 1, shuffle=True
)

