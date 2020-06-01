
import keras
from keras.utils.vis_utils import plot_model
#sudo apt-get install graphviz
#pip3 install pydotplus

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



x5 = keras.layers.convolutional.AveragePooling2D((7,7))(added4_1)
x5 = keras.layers.core.Flatten()(x5)
x5 = keras.layers.Dense(1000, activation='softmax')(x5)


model = keras.models.Model(input, x5)
model.summary()

plot_model(model,to_file='/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/resnet18.png', show_shapes= True)
