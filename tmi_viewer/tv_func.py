#!/usr/bin/env python

#    Various functions for tmi_viewer
#    Copyright (C) 2016  Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import sys
import numpy as np
import warnings
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
from scipy import ndimage
import scipy.misc as misc
from scipy.special import erf
import matplotlib.cbook
from skimage import filters
if 'QT_API' not in os.environ:
	os.environ['QT_API'] = 'pyqt'
try:
	from mayavi import mlab
except:
	print("Trying pyside")
	os.environ['QT_API'] = 'pyside'
	from mayavi import mlab


# makes sure that endianess is correct for the system
def check_byteorder(np_array):
	if sys.byteorder == 'little':
		sys_bo = '<'
	elif sys.byteorder == 'big':
		sys_bo = '>'
	else:
		pass
	if np_array.dtype.byteorder != sys_bo:
		np_array = np_array.byteswap().newbyteorder()
	return np_array


# applies the affine to the scalar field coordinates
def apply_affine_to_scalar_field(data, affine):
	data = np.array(data)
	if data.ndim == 4: # double check
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:,:,:,0]
	size_x, size_y, size_z = data.shape
	x,y,z = np.where(data!=55378008)
	coord = np.column_stack((x,y))
	coord = np.column_stack((coord,z))
	coord_array = nib.affines.apply_affine(affine, coord)
	xi = coord_array[:,0].reshape(size_x, size_y, size_z)
	yi = coord_array[:,1].reshape(size_x, size_y, size_z)
	zi = coord_array[:,2].reshape(size_x, size_y, size_z)

	src = mlab.pipeline.scalar_field(xi, yi, zi, data)
	return src

# applies the affine to the scalar field coordinates
def apply_affine_to_contour3d(data, affine, lthresh, hthresh, name, contours = 15, opacity = 0.7):
	data = np.array(data)
	if data.ndim == 4: # double check
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:,:,:,0]
	size_x, size_y, size_z = data.shape
	x,y,z = np.where(data!=55378008)
	coord = np.column_stack((x,y))
	coord = np.column_stack((coord,z))
	coord_array = nib.affines.apply_affine(affine, coord)
	xi = coord_array[:,0].reshape(size_x, size_y, size_z)
	yi = coord_array[:,1].reshape(size_x, size_y, size_z)
	zi = coord_array[:,2].reshape(size_x, size_y, size_z)
	src = mlab.contour3d(xi, yi, zi, data,
		vmin = lthresh,
		vmax = hthresh,
		opacity = opacity,
		name = name,
		contours=contours)
	return src

# returns the non-empty range
def nonempty_coordinate_range(data, affine):
	nonempty = np.argwhere(data!=0)
	nonempty_native = nib.affines.apply_affine(affine, nonempty)
	x_minmax = np.array((nonempty_native[:,0].min(), nonempty_native[:,0].max()))
	y_minmax = np.array((nonempty_native[:,1].min(), nonempty_native[:,1].max()))
	z_minmax = np.array((nonempty_native[:,2].min(), nonempty_native[:,2].max()))
	return (x_minmax,y_minmax,z_minmax)


# linear function look-up tables
def linear_cm(c0,c1,c2 = None):
	c_map = np.zeros((256,3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128,i] = np.linspace(c0[i],c1[i],128)
			c_map[127:256,i] = np.linspace(c1[i],c2[i],129)
	else:
		for i in range(3):
			c_map[:,i] = np.linspace(c0[i],c1[i],256)
	return c_map


# log function look-up tables
def log_cm(c0,c1,c2 = None):
	c_map = np.zeros((256,3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128,i] = np.geomspace(c0[i] + 1,c1[i] + 1,128)-1
			c_map[127:256,i] = np.geomspace(c1[i] + 1,c2[i] + 1,129)-1
	else:
		for i in range(3):
			c_map[:,i] = np.geomspace(c0[i] + 1,c1[i] + 1,256)-1
	return c_map


# error function look-up tables
def erf_cm(c0,c1,c2 = None):
	c_map = np.zeros((256,3))
	weights = erf(np.linspace(0,3,255))
	if c2 is not None:
		for i in range(3):
			c_map[0:128,i] = erf(np.linspace(3*(c0[i]/255),3*(c1[i]/255),128)) * 255
			c_map[127:256,i] = erf(np.linspace(3*(c1[i]/255),3*(c2[i]/255),129)) * 255
	else:
		for i in range(3):
			#c_map[:,i] = erf(np.linspace(0,3,256)) * np.linspace(c0[i], c1[i], 256)
			c_map[:,i] = erf(np.linspace(3*(c0[i]/255),3*(c1[i]/255),256)) * 255 
	return c_map


# display the luts included in matplotlib and the customs luts from tmi_viewer
def display_matplotlib_luts():
	# Adapted from https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html

	# This example comes from the Cookbook on www.scipy.org. According to the
	# history, Andrew Straw did the conversion from an old page, but it is
	# unclear who the original author is.

	plt.switch_backend('Qt4Agg')
	warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

	a = np.linspace(0, 1, 256).reshape(1,-1)
	a = np.vstack((a,a))

	maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
	maps.append('red-yellow') # custom maps 
	maps.append('blue-lightblue')
	maps.append('green-lightgreen')
	maps.append('tm-breeze')
	maps.append('tm-sunset')
	maps.append('tm-broccoli')
	maps.append('tm-octopus')
	maps.append('tm-storm')
	maps.append('tm-flow')
	maps.append('tm-logBluGry')
	maps.append('tm-logRedYel')
	maps.append('tm-erfRGB')
	nmaps = len(maps) + 1

	fig = plt.figure(figsize=(8,12))
	fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
	for i,m in enumerate(maps):
		ax = plt.subplot(nmaps, 1, i+1)
		plt.axis("off")
		if m == 'red-yellow':
			cmap_array = linear_cm([255,0,0],[255,255,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'blue-lightblue':
			cmap_array = linear_cm([0,0,255],[0,255,255]) / 255
#			cmap_array = np.array(( np.zeros(256), np.linspace(0,255,256), (np.ones(256)*255) )).T / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'green-lightgreen':
			cmap_array = linear_cm([0,128,0],[0,255,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-breeze':
			cmap_array = linear_cm([199,233,180],[65,182,196],[37,52,148]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-sunset':
			cmap_array = linear_cm([255,255,51],[255,128,0],[204,0,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-storm':
			cmap_array = linear_cm([0,153,0],[255,255,0],[204,0,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-flow':
			cmap_array = log_cm([51,51,255],[255,0,0],[255,255,255]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-logBluGry':
			cmap_array = log_cm([0,0,51],[0,0,255],[255,255,255]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-logRedYel':
			cmap_array = log_cm([102,0,0],[200,0,0],[255,255,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-erfRGB':
			cmap_array = erf_cm([255,0,0],[0,255,0], [0,0,255]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-broccoli':
			cmap_array = linear_cm([204,255,153],[76,153,0], [0,102,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-octopus':
			cmap_array = linear_cm([255,204,204],[255,0,255],[102,0,0]) / 255
			plt.imshow(a, aspect='auto', cmap=colors.ListedColormap(cmap_array,m), origin='lower')
		else:
			plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
		pos = list(ax.get_position().bounds)
		fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
	plt.show()


# Get RGBA colormap [uint8, uint8, uint8, uint8]
def get_cmap_array(lut, background_alpha = 255, image_alpha = 1.0, zero_lower = True, zero_upper = False, base_color = [227,218,201,0], c_reverse = False):
	base_color[3] = background_alpha
	if lut.endswith('_r'):
		c_reverse = lut.endswith('_r')
		lut = lut[:-2]
	# make custom look-up table
	if (str(lut) == 'r-y') or (str(lut) == 'red-yellow'):
		cmap_array = np.column_stack((linear_cm([255,0,0],[255,255,0]), (255 * np.ones(256) * image_alpha)))
	elif (str(lut) == 'b-lb') or (str(lut) == 'blue-lightblue'):
		cmap_array = np.column_stack((linear_cm([0,0,255],[0,255,255]), (255 * np.ones(256) * image_alpha)))
	elif (str(lut) == 'g-lg') or (str(lut) == 'green-lightgreen'):
		cmap_array = np.column_stack((linear_cm([0,128,0],[0,255,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-breeze':
		cmap_array = np.column_stack((linear_cm([199,233,180],[65,182,196],[37,52,148]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-sunset':
		cmap_array = np.column_stack((linear_cm([255,255,51],[255,128,0],[204,0,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-broccoli':
		cmap_array = np.column_stack((linear_cm([204,255,153],[76,153,0],[0,102,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-octopus':
		cmap_array = np.column_stack((linear_cm([255,204,204],[255,0,255],[102,0,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-storm':
		cmap_array = np.column_stack((linear_cm([0,153,0],[255,255,0],[204,0,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-flow':
		cmap_array = np.column_stack((log_cm([51,51,255],[255,0,0],[255,255,255]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-logBluGry':
		cmap_array = np.column_stack((log_cm([0,0,51],[0,0,255],[255,255,255]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-logRedYel':
		cmap_array = np.column_stack((log_cm([102,0,0],[200,0,0],[255,255,0]),(255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-erfRGB':
		cmap_array = np.column_stack((erf_cm([255,0,0],[0,255,0], [0,0,255]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-white':
		cmap_array = np.column_stack((linear_cm([255,255,255],[255,255,255]), (255 * np.ones(256) * image_alpha)))
	else:
		try:
			cmap_array = eval('plt.cm.%s(np.arange(256))' % lut)
			cmap_array[:,3] = cmap_array[:,3] = image_alpha
		except:
			print("Error: Lookup table '%s' is not recognized." % lut)
			print("The lookup table can be red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html)")
			sys.exit()
		cmap_array *= 255
	if c_reverse:
		cmap_array = cmap_array[::-1]
	if zero_lower:
		cmap_array[0] = base_color
	if zero_upper:
		cmap_array[-1] = base_color
	return cmap_array


# Remove black from png
def correct_image(img_name, rotate = None, b_transparent = True, flip = False):
	img = misc.imread(img_name)
	if b_transparent:
		if img_name.endswith('.png'):
			rows = img.shape[0]
			columns = img.shape[1]
			if img.shape[2] == 3:
				img_flat = img.reshape([rows * columns, 3])
			else:
				img_flat = img.reshape([rows * columns, 4])
				img_flat = img_flat[:,:3]
			alpha = np.zeros([rows*columns, 1], dtype=np.uint8)
			alpha[img_flat[:,0]!=0] = 255
			alpha[img_flat[:,1]!=0] = 255
			alpha[img_flat[:,2]!=0] = 255
			img_flat = np.column_stack([img_flat, alpha])
			img = img_flat.reshape([rows, columns, 4])
	if rotate is not None:
		img = ndimage.rotate(img, float(rotate))
	if flip:
		img = img[:,::-1,:]
	misc.imsave(img_name, img)

# add coordinates to the image slices
def add_text_to_img(image_file, add_txt, opacity = 200, color = [0,0,0]):
	from PIL import Image, ImageDraw, ImageFont
	base = Image.open(image_file).convert('RGBA')
	txt = Image.new('RGBA', base.size, (255,255,255,0))
	numpixels = base.size[0]
	fnt_size = int(numpixels / 16) # scale the font
	fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', fnt_size)
	start = int(numpixels - (.875 * numpixels))
	stop = int(.875 * numpixels)

	d = ImageDraw.Draw(txt)
	d.text((start,start), str(add_txt), font=fnt, fill=(color[0],color[1],color[2],opacity))
	out = Image.alpha_composite(base, txt)
	mpimg.imsave(image_file, np.array(out))

# crop and concatenate images
def concate_images(basename, num, clean=False, numpixels = 400):
	start = int(numpixels - (.875 * numpixels))
	stop = int(.875 * numpixels)
	for i in range(num):
		if i == 0:
			outpng = mpimg.imread("%s_0.png" % basename)[start:stop,start:stop,:]
		else:
			tempng = mpimg.imread("%s_%d.png" % (basename, i))[start:stop,start:stop,:]
			outpng = np.concatenate((outpng,tempng),1)
		if i == (num-1):
			mpimg.imsave('%ss.png' % basename, outpng)
	if clean:
		for i in range(num):
			os.remove("%s_%d.png" % (basename, i))


# mask image and draw the outline
def draw_outline(img_png, mask_png, outline_color = [1,0,0,1]):
	from scipy.ndimage.morphology import binary_erosion
	img = mpimg.imread(img_png)
	mask = mpimg.imread(mask_png)
	#check mask
	mask[mask[:,:,3] != 1] = [0,0,0,0]
	mask[mask[:,:,3] == 1] = [1,1,1,1]
	mask[mask[:,:,0] == 1] = [1,1,1,1]
	index = (mask[:,:,0] == 1)
	ones_arr = index *1
	m = ones_arr - binary_erosion(ones_arr)
	index = (m[:,:] == 1)
	img[index] = outline_color
	os.remove(mask_png)
	mpimg.imsave(img_png, img)


# various methods for choosing thresholds automatically
def autothreshold(data, threshold_type = 'yen', z = 2.3264):
	if threshold_type.endswith('_p'):
		data = data[data>0]
	else:
		data = data[data!=0]
	if data.size == 0:
		print("Warning: the data array is empty. Auto-thesholding will not be performed")
		return 0, 0
	else:
		if (threshold_type == 'otsu') or (threshold_type == 'otsu_p'):
			lthres = filters.threshold_otsu(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
			# Otsu N (1979) A threshold selection method from gray-level histograms. IEEE Trans. Sys., Man., Cyber. 9: 62-66.
		elif (threshold_type == 'li')  or (threshold_type == 'li_p'):
			lthres = filters.threshold_li(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
			# Li C.H. and Lee C.K. (1993) Minimum Cross Entropy Thresholding Pattern Recognition, 26(4): 617-625
		elif (threshold_type == 'yen') or (threshold_type == 'yen_p'):
			lthres = filters.threshold_yen(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
			# Yen J.C., Chang F.J., and Chang S. (1995) A New Criterion for Automatic Multilevel Thresholding IEEE Trans. on Image Processing, 4(3): 370-378.
		elif threshold_type == 'zscore_p':
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
			if lthres < 0:
				lthres = 0.001
		else:
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
		if uthres > data.max(): # for the rare case when uthres is larger than the max value
			uthres = data.max()
		return lthres, uthres


# makes a webpage of slices
def make_slice_html(outname, coordinates, iv, write_coordinates = True):
	if not os.path.exists(".%s" % outname):
		os.mkdir(".%s" % outname)
	os.system("mv ?_Slices.png .%s/" % outname)
	os.system("mv ?_colorbar.png .%s/" % outname)
	with open(outname, "wb") as o:
		o.write("<!DOCTYPE HTML>\n")
		o.write("<html lang = 'en'>\n")
		o.write("<head>\n")
		o.write("  <title>TMI Snapshots</title>\n")
		o.write("  <meta charset = 'UTF-8' />\n")
		o.write("  <style>\n")
		o.write("      p {\n")
		o.write("        font-size: 1vw;\n")
		o.write("        padding: 5px;\n")
		o.write("      }\n")
		o.write("      ul {\n")
		o.write("        white-space: nowrap;\n")
		o.write("      }\n")
		o.write("      ul, li {\n")
		o.write("        list-style: none;\n")
		o.write("        display: inline;\n")
		o.write("      }\n")
		o.write("  </style>\n")
		o.write("</head>\n")
		o.write("<body>\n")
		o.write("  <h1>X Axis </h1>\n")
		if write_coordinates:
			o.write("    <p><span> X = %s </span>" % ', X = '.join([ '%.0f' % elem for elem in coordinates[:,0] ]))
			o.write("    </p>")
		o.write("    <a href='.%s/X_Slices.png'>\n" % outname)
		o.write("    <img src = '.%s/X_Slices.png' width='100%%'>\n" % outname)
		o.write("    </a>\n")
		o.write("  <h1> Y Axis </h1>\n")
		if write_coordinates:
			o.write("    <p><centre> Y = %s</centre>" % ', Y = '.join([ '%.0f' % elem for elem in coordinates[:,1] ]))
			o.write("    </p>")
		o.write("    <a href='.%s/Y_Slices.png'>\n" % outname)
		o.write("    <img src = '.%s/Y_Slices.png' width='100%%'>\n" % outname)
		o.write("    </a>\n")
		o.write("  <h1>Z Axis </h1>\n")
		if write_coordinates:
			o.write("    <p>Z = %s " % ', Z = '.join([ '%.0f' % elem for elem in coordinates[:,2] ]))
			o.write("    </p>")
		o.write("    <a href='.%s/Z_Slices.png'>\n" % outname)
		o.write("    <img src = '.%s/Z_Slices.png' width='100%%'>\n" % outname)
		o.write("    </a>\n")
		o.write("    <h1> Color Bar(s) </h1>\n")
		o.write("    <ul>\n")
		for i in range(len(iv)):
			o.write("        <a href='.%s/%d_colorbar.png'>\n" % (outname, i))
			o.write("        <li><img src='.%s/%d_colorbar.png' width='auto' height='200'></li>\n" % (outname, i))
			o.write("        </a>\n")
		o.write("    </ul>\n")
		o.write("    <a href='.%s/settings'>\n" % outname)
		o.write("    <p>tm_slices settings</p>\n")
		o.write("    </a>\n")
		o.write("    <p>Made with TFCE_mediation and tmi_viewer</p>\n")
		o.write("</body>\n")
		o.write("</html>\n")
		o.close()


# write tmi_viewer html
def make_html(title, basename, lut, outname): # outdated
	with open(outname, "wb") as o:
		o.write("<!DOCTYPE HTML>\n")
		o.write("<html lang = 'en'>\n")
		o.write("<head>\n")
		o.write("  <title>TMI Snapshots</title>\n")
		o.write("  <meta charset = 'UTF-8' />\n")
		o.write("</head>\n")
		o.write("<body>\n")
		o.write("  <h1>%s</h1>\n" % title)
		o.write("   <ul>\n")
		o.write("    <a href='%s_anterior.png'>\n" % basename)
		o.write("    <img src = '%s_anterior.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:80px;' > Anterior</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_posterior.png'>\n" % basename)
		o.write("    <img src = '%s_posterior.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:280px;'> Posterior</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_left.png'>\n" % basename)
		o.write("    <img src = '%s_left.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:500px;'> Left</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_right.png'>\n" % basename)
		o.write("    <img src = '%s_right.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:700px;'> Right</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_superior.png'>\n" % basename)
		o.write("    <img src = '%s_superior.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:900px;'> Superior</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_inferior.png'>\n" % basename)
		o.write("    <img src = '%s_inferior.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:1110px;'> Inferior</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_isometric.png'>\n" % basename)
		o.write("    <img src = '%s_isometric.png' height='200' width='200'>\n" % basename)
		o.write("    <h1 style='position:absolute; top:40px; left:1300px;'> Isometric</h1>\n")
		o.write("    </a>\n")
		o.write("    <a href='%s_colorbar.png'>\n" % lut)
		o.write("    <img src = '%s_colorbar.png' height='200' width='40'>\n" % lut)
		o.write("    </a>\n")
		o.write("   </ul>\n")
		o.write("</body>\n")
		o.write("</html>\n")
		o.close()

# saves the output dictionary of argparse to a file
def write_dict(filename, outnamespace):
	with open(filename, "wb") as o:
		for k in outnamespace.__dict__:
			if outnamespace.__dict__[k] is not None:
				o.write("%s : %s\n" % (k, outnamespace.__dict__[k]))
		o.close()
