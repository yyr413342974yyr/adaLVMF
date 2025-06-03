import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, UpSampling2D, Concatenate, LeakyReLU, Dropout, BatchNormalization, Lambda, Conv2DTranspose, ReLU
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras



# from blurpool import BlurPool2D

def UpSampling2DBilinear(size):
	#from stackoverflow.com/questions/44186042/keras-methods-to-enlarge-spartial-dimension-of-the-layer-output-blob
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

def combineChannels(j, i):
	#remove channel i and add it to channel j
	assert j < i
	def combiner(x):
		si = x[...,i:i+1]
		sj = x[...,j:j+1]
		y = tf.concat([x[...,:j], si+sj, x[...,j+1:i], x[...,i+1:]], axis=-1)
		return y
	return combiner

def makeUNet(input_shape=(64,64,1), down_step=3, initial_channels=16, do_batchnorm=False, do_dropout=False, dropout_rate=0.1, act='relu', out_channels=3):
	#has 31,377,988 params with down_step=5, initial_channels=64

	input_layer = h = Input(input_shape)

	h_s = []
	for j in range(down_step):

		if j != 0:
			h_s.append(h)
			h = MaxPooling2D()(h)

		for i in range(2):
			h = Conv2D(initial_channels*2**j,3,padding='same', kernel_initializer='he_normal')(h)
			if do_batchnorm:
				h = BatchNormalization()(h)
			if act == 'relu':
				h = ReLU()(h)
			else:
				h = LeakyReLU()(h)
			if do_dropout:
				h = Dropout(dropout_rate)(h)
			
	for j in reversed(range(down_step-1)):

		h = UpSampling2D()(h)
		h = Concatenate()([h,h_s.pop()])

		for i in range(2):
			h = Conv2D(initial_channels*2**j,3,padding='same', kernel_initializer='he_normal')(h)
			if do_batchnorm:
				if j != 0:
					h = BatchNormalization()(h)
			if act == 'relu':
				h = ReLU()(h)
			else:
				h = LeakyReLU()(h)
			if do_dropout:
				if j != 0:
					h = Dropout(dropout_rate)(h)

	out1 = Conv2D(out_channels,1,activation='sigmoid',padding='same')(h)

	# out1 = Conv2D(4,1,activation='softmax',padding='same')(h)
	# output_1 = Lambda(combineChannels(0,1),name='pathologies')(out1)
	# output_2 = Lambda(combineChannels(2,3),name='anatomy')(out1)

	# output_1 = Conv2D(3,1,activation='softmax',padding='same',name='pathologies')(h)
	# output_2 = Conv2D(3,1,activation='softmax',padding='same',name='anatomy')(h)

	model = Model(input_layer, out1)#[output_1, output_2])

	# from keras.utils import plot_model
	# plot_model(model, to_file='model.png')
	# model.summary()

	return model


def makeRueckertNet(input_shape=(64,64,1), down_step=5, initial_channels=16, up_method=0, out_channels=3):

	assert keras.backend.image_data_format() == 'channels_last'
	assert len(input_shape) == 3 #input images should be HxWxC

	input_layer = h = Input(input_shape)

	h = Conv2D(initial_channels, 3, padding='same', kernel_initializer='he_normal')(h)
	h = BatchNormalization()(h)
	h = ReLU()(h)

	finals = []
	for i in range(down_step):

		h = Conv2D(initial_channels*2**i, 3, padding='same', kernel_initializer='he_normal')(h)
		h = BatchNormalization()(h)
		h = ReLU()(h)

		if i > 1:
			h = Conv2D(initial_channels*2**i, 3, padding='same', kernel_initializer='he_normal')(h)
			h = BatchNormalization()(h)
			h = ReLU()(h)

		e = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(h)
		e = BatchNormalization()(e)
		e = ReLU()(e)

		if up_method == 0:
			e = Conv2DTranspose(32, 4, strides=2**i, padding='same', kernel_initializer='he_normal')(e)
		if up_method == 1:
			e = Conv2DTranspose(32, 3, strides=2**i, padding='same', kernel_initializer='he_normal')(e)
		if up_method == 2:
			e = UpSampling2D(2**i)(e)
			e = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(e)
		if up_method > 2: #methods using bi-linear upsampling
			size = (e.shape[1]*2**i, e.shape[2]*2**i)
			e = UpSampling2DBilinear(size)(e)
			if up_method == 3:
				e = Conv2D(32, 3, padding='same', dilation_rate=2**i, kernel_initializer='he_normal')(e)
			if up_method == 4:
				pass
			if up_method == 5:
				e = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(e)
		e = BatchNormalization()(e)
		e = ReLU()(e)
		finals.append(e)

		if i != down_step-1:
			h = Conv2D(initial_channels*2**(i+1), 3, strides=2, padding='same', kernel_initializer='he_normal')(h)
			h = BatchNormalization()(h)
			h = ReLU()(h)

	h = Concatenate(axis=-1)(finals)

	h = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(h)
	h = BatchNormalization()(h)
	h = ReLU()(h)

	h = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(h)
	h = BatchNormalization()(h)
	h = ReLU()(h)

	out1 = Conv2D(out_channels,1,activation='sigmoid',padding='same')(h)
	# output_1 = Lambda(combineChannels(0,1),name='pathologies')(out1)
	# output_2 = Lambda(combineChannels(2,3),name='anatomy')(out1)

	# model = Model(input_layer, [output_1, output_2])

	model = Model(input_layer, out1)

	model.summary()

	return model

def makeVGG16(input_shape=(64,64,1), pretrained_weights=None):
	#has 15,598,377 params

	assert keras.backend.image_data_format() == 'channels_last'
	assert len(input_shape) == 3 #input images should be HxWxC
	assert input_shape[0] == input_shape[1] #input images should be square

	h = input_layer = Input(input_shape)

	#make the input have three channels if it doesn't already:
	if input_shape[-1] != 3:
		h = Conv2D(3, 1, padding='same')(h)

	weights = 'imagenet' if pretrained_weights else None
	vgg = keras.applications.vgg16.VGG16(include_top=True, input_shape=h.shape[1:], weights=weights)
#	vgg = keras.applications.vgg16.VGG16(include_top=True, input_shape=h._shape_tuple()[1:], weights=weights)
	h = vgg(h)
	# h = BatchNormalization()(h)
	# h = Dense(100)(h)
	# h = BatchNormalization()(h)
	# h = ReLU()(h)
	output_layer = Dense(1)(h)
	model = Model(input_layer, output_layer)

	return model



def makeRueckertNetVGG16(input_shape=(64,64,1), pretrained_weights=True, out_channels=3):
	#has 15,598,377 params

	assert keras.backend.image_data_format() == 'channels_last'
	assert len(input_shape) == 3 #input images should be HxWxC
	assert input_shape[0] == input_shape[1] #input images should be square

	h = input_layer = Input(input_shape)

	#make the input have three channels if it doesn't already:
	if input_shape[-1] != 3:
		h = Conv2D(3, 1, padding='same')(h)

	weights = 'imagenet' if pretrained_weights else None
	model = keras.applications.vgg16.VGG16(include_top=False, input_shape=h.shape[1:], weights=weights)
#	model = keras.applications.vgg16.VGG16(include_top=False, input_shape=h._shape_tuple()[1:], weights=weights)

	#make encoder and get layers for concat:
	for_skip_names, for_skips = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'], []
	layer_outputs = []
	for layer in model.layers:
		h = layer(h)
		if layer.name in for_skip_names:
			# if h.shape[1] < input_shape[1]:
			strides = input_shape[1] // int(h.shape[1])
			e = Conv2DTranspose(32, 4, strides=strides, padding='same')(h)
			for_skips.append(e)

	h = Concatenate(axis=-1)(for_skips)

	h = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(h)
	h = BatchNormalization()(h)
	h = ReLU()(h)

	h = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(h)
	h = BatchNormalization()(h)
	h = ReLU()(h)

	output_layer = Conv2D(out_channels, 1, activation='sigmoid', padding='same')(h)

	model = Model(input_layer, output_layer)

	# model.summary()
	# from keras.utils import plot_model
	# plot_model(model, to_file='model.png', show_shapes=True)

	return model


def makeTernausNet16(input_shape=(64,64,1), pretrained_weights=True, out_channels=3, shift_invariant_mod=False, use_dropout=False):
	#has 31,337,994 params

	assert keras.backend.image_data_format() == 'channels_last'
	assert len(input_shape) == 3 #input images should be HxWxC
	assert input_shape[0] == input_shape[1] #input images should be square

	h = input_layer = Input(input_shape)

	#make the input have three channels if it doesn't already:
	if input_shape[-1] != 3:
		h = Conv2D(3, 1, padding='same')(h)

	weights = 'imagenet' if pretrained_weights else None
	model = keras.applications.vgg16.VGG16(include_top=False, input_shape=h.shape[1:], weights=weights)
#	model = keras.applications.vgg16.VGG16(include_top=False, input_shape=h._shape_tuple()[1:], weights=weights)

	#make encoder and get inputs for skip connections:
	for_skip_names, for_skips = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'], []
	layer_outputs = []
	for layer in model.layers:

		if layer.name.split('_')[1] == 'pool' and shift_invariant_mod == True:
			h = MaxPooling2D(strides=1, padding='same')(h)
			# h = BlurPool2D()(h)
		else:
			h = layer(h)
		if layer.name in for_skip_names:
			for_skips.append(h)

	#make decoder using skip connections:
	dbs = [256,256,256,64,32]
	for i in dbs:
		h = Conv2D(i*2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(h)
		h = Dropout(0.5)(h)
		h = Conv2DTranspose(i, 4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(h)
		h = Concatenate(axis=-1)([h,for_skips.pop()])

	h = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(h)
	output_layer = Conv2D(4, 1, padding='same', activation='softmax')(h)
	output_layer = Lambda(lambda x : x[...,:3])(output_layer)
	# output_layer = Conv2D(out_channels, 1, padding='same', activation='sigmoid')(h)

	model = Model(input_layer, output_layer)

	return model

if __name__ == '__main__':

	makeTernausNet16()
	# makeUNet()
	# makeRueckertNet()
	# makeRueckertNetVGG16()

