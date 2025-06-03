import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw 


def addLabels(arr,x,y,dx,dy,texts,font_size=16,font_color=(255,255,255), blackbox=True, blackbox_color='black'):
	'''
	adds labels (i.e. some text) to an image
	
	argument:
		arr : the image to add text to, should be a numy array
		x,y : the (bottom left) position of the first label text (in pixels, 0,0 = left,top)
		dx,dy : the ammount to shift (in pixels) between labels
		texts : a list of strings to be used as the labels

		optional:
		font_size : the font size (height in pixels I think)
		font_color : the font color as an RGB tuple, should be a tuple of 3 ints in [0,255]

	returns:
		the image (as a numpy array) with the text added
	'''

	img = Image.fromarray(arr)
	draw = ImageDraw.Draw(img)

	# if os.name == 'nt': #windows
	# 	font_path = "arial.ttf"
	# elif os.name == 'posix': #unix
	# 	font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
	# else:
	# 	print('not sure on the OS, trying font_path="arial.ttf"')
	font_path = "font/FreeMono.ttf"

	font = ImageFont.truetype(font_path, font_size)

	for i, text in enumerate(texts):
		if blackbox:
			w, h = font.getsize(text)
			draw.rectangle((x+i*dx, y+i*dy, x+i*dx + w, y+i*dy + h), fill='black')
		draw.text((x+i*dx,y+i*dy), text, font_color, font=font)

	return np.asarray(img)