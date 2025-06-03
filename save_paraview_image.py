# 6 = 0,0
# 5 = 2,0
# 4 = 2,1
# 3 = 0,3
# 2 = 1,0

from paraview.simple import *
from imageio import imread
from imageio import imwrite as imsave
import numpy as np
import meshio
import os

from matplotlib import pyplot as plt

def saveMeshImage(vtk_file, button, filename, 
	cut_on_axis=None, 
	second_vtk=None, 
	magnification=1,
	show_error_as_color=False,
	show_image_plane=False,
	clip_shift=0.0
	):

	assert button in [1,2,3,4,5,6]

	# plane = Plane(Origin=[-64,0,8], Point1=[64,0,8], Point2=[-64,0,-120])

	if show_image_plane:
		if button == 1:
			plane = Plane(Origin=[64,-64,8], Point1=[64,64,8], Point2=[64,-64,-120])
		if button == 2:
			plane = Plane(Origin=[-64,-64,8], Point1=[-64,64,8], Point2=[-64,-64,-120])
		if button == 3:
			plane = Plane(Origin=[-64,64,8], Point1=[64,64,8], Point2=[-64,64,-120])
		if button == 4:
			plane = Plane(Origin=[-64,-64,8], Point1=[64,-64,8], Point2=[-64,-64,-120])
		if button == 5:
			plane = Plane(Origin=[-64,-64,8], Point1=[-64,64,8], Point2=[64,-64,8])
		if button == 6:
			plane = Plane(Origin=[-64,-64,-128], Point1=[-64,64,-128], Point2=[64,-64,-128])
		Show(plane)
		dpp = GetDisplayProperties(plane)
		dpp.DiffuseColor = [0, 0, 0]

		ResetCamera()

	if second_vtk != None and show_error_as_color == True:

		mesh1 = meshio.read(vtk_file)
		mesh2 = meshio.read(second_vtk)

		mesh_points_1 = mesh1.points
		mesh_points_2 = mesh2.points

		point_distances = np.sum( (mesh_points_1-mesh_points_2)**2, axis=1)**0.5

		tmp_file_1 = vtk_file[-4]+'_with_err_tmp.vtk'
		mesh1.point_data['dist'] = point_distances
		meshio.write(tmp_file_1, mesh1)

		tmp_file_2 = second_vtk[-4]+'_with_err_tmp.vtk'
		mesh2.point_data['dist'] = point_distances
		meshio.write(tmp_file_2, mesh2)

		orig_reader = LegacyVTKReader(FileNames=tmp_file_1)
		if second_vtk != None:
			orig_reader2 = LegacyVTKReader(FileNames=tmp_file_2)

		os.remove(tmp_file_1)
		os.remove(tmp_file_2)

	else:
		orig_reader = LegacyVTKReader(FileNames=vtk_file)
		if second_vtk != None:
			orig_reader2 = LegacyVTKReader(FileNames=second_vtk)

	if cut_on_axis == None:
		reader = orig_reader
		if second_vtk != None:
			reader2 = orig_reader2
	else:
		reader = Clip(orig_reader)
		if second_vtk != None:
			reader2 = Clip(orig_reader2)

	# reader.ClipType.Normal = [0,0,1]
	# reader.ClipType.Origin = [clip_shift,0.0,-30.0]

	rdsp = Show(reader)
	if second_vtk != None and not show_error_as_color:
		Show(reader2)

		dp = GetDisplayProperties(reader)
		dp.DiffuseColor = [0, 170/255, 1]
		dp.Opacity = 0.5

		dp2 = GetDisplayProperties(reader2)
		dp2.DiffuseColor = [1, 0, 127/255]
	elif second_vtk != None and show_error_as_color:

		dp = GetDisplayProperties(reader)
		dp.DiffuseColor = [0, 170/255, 1]
		dp.Opacity = 0.5
	else:
		pass
		# dp = GetDisplayProperties(reader)
		# dp.ColorArrayName = 'dist'
		# ColorBy(rdsp, ('dist'))

	camera = GetActiveCamera()

	view = GetActiveView()
	view.OrientationAxesVisibility = 0
	view.CameraParallelProjection = 1
	view.Background = [1,1,1]

	# print( dir(view) )
	# sys.exit()

	# Render()
	# fn = filename[:-4]+'_pre.png'
	# WriteImage(fn, Magnification=magnification)


	if button == 1:
		a,e=90,180
		camera.Azimuth(a)
		camera.Elevation(e)
		camera.Roll(90)
		Render()
	if button == 2:
		a,e=90,0
		camera.Azimuth(a)
		camera.Elevation(e)
		camera.Roll(270)
		Render()
	if button == 3:
		a,e=0,270
		camera.Azimuth(a)
		camera.Elevation(e)
		Render()
	if button == 4:
		a,e=180,90
		camera.Azimuth(a)
		camera.Elevation(e)
		Render()
	if button == 5:
		a,e=180,0
		camera.Azimuth(a)
		camera.Elevation(e)
		Render()
	if button == 6:
		Render()

	WriteImage(filename, Magnification=magnification)

	if button == 1:
		a,e=90,180
		camera.Roll(270)
		camera.Elevation(360-e)
		camera.Azimuth(360-a)
	if button == 2:
		a,e=90,0
		camera.Roll(90)
		camera.Elevation(360-e)
		camera.Azimuth(360-a)
	if button == 3:
		a,e=0,270
		camera.Elevation(360-e)
		camera.Azimuth(360-a)
	if button == 4:
		a,e=180,90
		camera.Elevation(360-e)
		camera.Azimuth(360-a)
	if button == 5:
		a,e=180,0
		camera.Elevation(360-e)
		camera.Azimuth(360-a)
	if button == 6:
		pass

	# Render()
	# fn = filename[:-4]+'_post.png'
	# WriteImage(fn, Magnification=magnification)

	Hide(reader)
	# Delete(reader)
	if second_vtk != None:
		Hide(reader2)
		# Delete(reader2)

	if show_image_plane:
		Hide(plane)
		Delete(plane)

		#trim white border and resave:
		img = imread(filename)
		img_mean = np.mean(img, axis=2)
		rows = np.any(np.where(img_mean!=255,1,0), axis=1)
		cols = np.any(np.where(img_mean!=255,1,0), axis=0)
		rmin, rmax = np.where(rows)[0][[0, -1]]
		cmin, cmax = np.where(cols)[0][[0, -1]]
		imsave(filename, img[cmin:cmax, rmin:rmax])

	return



if __name__ == '__main__':

	saveMeshImage("tmp.vtk", button=1, filename='views/button=1.png')
	saveMeshImage("tmp.vtk", button=2, filename='views/sliced.png', cut_on_axis=True)
	saveMeshImage("tmp.vtk", button=1, filename='views/two_images.png', second_vtk="ShapeModel/Mean/LV_mean.vtk")
	saveMeshImage("tmp.vtk", button=2, filename='views/two_images_sliced.png', second_vtk="ShapeModel/Mean/LV_mean.vtk", cut_on_axis=True)

	