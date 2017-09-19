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

from __future__ import division
import os
import sys
import numpy as np
import warnings
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
from scipy import ndimage
import matplotlib.cbook
from skimage import filters
if 'QT_API' not in os.environ:
	os.environ['QT_API'] = 'pyqt'
try:
	from mayavi import mlab
except:
	print "Trying pyside"
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
	size_x, size_y, size_z = data.shape
	x,y,z = np.where(data!=55378008)
	coord = np.column_stack((x,y))
	coord = np.column_stack((coord,z))
	coord_array = nib.affines.apply_affine(affine, coord)
	xi = coord_array[:,0].reshape(size_x, size_y, size_z) * np.sign(affine[0,0])
	yi = coord_array[:,1].reshape(size_x, size_y, size_z) * np.sign(affine[1,1])
	zi = coord_array[:,2].reshape(size_x, size_y, size_z) * np.sign(affine[2,2])
	src = mlab.pipeline.scalar_field(xi, yi, zi, data)
	return src


# returns the non-empty range
def nonempty_coordinate_range(data, affine):
	nonempty = np.argwhere(data!=0)
	nonempty_native = nib.affines.apply_affine(affine, nonempty)
	x_minmax = np.array((nonempty_native[:,0].min(), nonempty_native[:,0].max()))
	y_minmax = np.array((nonempty_native[:,1].min(), nonempty_native[:,1].max()))
	z_minmax = np.array((nonempty_native[:,2].min(), nonempty_native[:,2].max()))
	return (x_minmax,y_minmax,z_minmax)


# function for creating linear look-up tables
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


# function for creating log look-up tables
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
	maps.append(u'red-yellow') # custom maps 
	maps.append(u'blue-lightblue')
	maps.append(u'green-lightgreen')
	maps.append(u'tm-breeze')
	maps.append(u'tm-summer')
	maps.append(u'tm-storm')
	maps.append(u'tm-flow')
	maps.append(u'tm-logBluGry')
	maps.append(u'tm-logRedYel')
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
		elif m == 'tm-summer':
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
		else:
			plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
		pos = list(ax.get_position().bounds)
		fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
	plt.show()


# Get RGBA colormap [uint8, uint8, uint8, uint8]
def get_cmap_array(lut, alpha = 255, zero_lower = True, zero_upper = False, base_color = [227,218,201,0], c_reverse = False):
	base_color[3] = alpha
	if lut.endswith('_r'):
		c_reverse = lut.endswith('_r')
		lut = lut[:-2]
	# make custom look-up table
	if (str(lut) == 'r-y') or (str(lut) == 'red-yellow'):
		cmap_array = np.column_stack((linear_cm([255,0,0],[255,255,0]),np.ones(256)*255))
	elif (str(lut) == 'b-lb') or (str(lut) == 'blue-lightblue'):
		cmap_array =  np.column_stack((linear_cm([0,0,255],[0,255,255]),np.ones(256)*255))
	elif (str(lut) == 'g-lg') or (str(lut) == 'green-lightgreen'):
		cmap_array =  np.column_stack((linear_cm([0,128,0],[0,255,0]),np.ones(256)*255))
	elif str(lut) == 'tm-breeze':
		cmap_array =  np.column_stack((linear_cm([199,233,180],[65,182,196],[37,52,148]),np.ones(256)*255))
	elif str(lut) == 'tm-summer':
		cmap_array =  np.column_stack((linear_cm([255,255,51],[255,128,0],[204,0,0]),np.ones(256)*255))
	elif str(lut) == 'tm-storm':
		cmap_array =  np.column_stack((linear_cm([0,153,0],[255,255,0],[204,0,0]),np.ones(256)*255))
	elif str(lut) == 'tm-flow':
		cmap_array =  np.column_stack((log_cm([51,51,255],[255,0,0],[255,255,255]),np.ones(256)*255))
	elif str(lut) == 'tm-logBluGry':
		cmap_array =  np.column_stack((log_cm([0,0,51],[0,0,255],[255,255,255]),np.ones(256)*255))
	elif str(lut) == 'tm-logRedYel':
		cmap_array =  np.column_stack((log_cm([102,0,0],[200,0,0],[255,255,0]),np.ones(256)*255))
	elif str(lut) == 'tm-white':
		cmap_array =  np.column_stack((linear_cm([255,255,255],[255,255,255]),np.ones(256)*255))
	else:
		try:
			cmap_array = eval('plt.cm.%s(np.arange(256))' % lut)
		except:
			print "Error: Lookup table '%s' is not recognized." % lut
			print "The lookup table can be red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html)"
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
def correct_image(img_name, rotate = None, b_transparent = True):
	img = mpimg.imread(img_name)
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
			alpha.fill(1)
			alpha[np.equal([0,0,0], img_flat).all(1)] = [0]
			img_flat = np.column_stack([img_flat, alpha])
			img = img_flat.reshape([rows, columns, 4])
	if rotate is not None:
		img = ndimage.rotate(img, float(rotate))
	mpimg.imsave(img_name, img)


# crop and concatenate images
def concate_images(basename, num, clean=False):
	for i in range(num):
		if i == 0:
			outpng = mpimg.imread("%s_0.png" % basename)[50:350,50:350,:]
		else:
			tempng = mpimg.imread("%s_%d.png" % (basename, i))[50:350,50:350,:]
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
def autothreshold(data, threshold_type = 'otsu', z = 2.3264):
	if threshold_type.endswith('_p'):
		data = data[data>0]
	else:
		data = data[data!=0]
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
	else:
		lthres = data.mean() - (z*data.std())
		uthres = data.mean() + (z*data.std())
	return lthres, uthres


# makes a webpage of slices
def make_slice_html(outname, coordinates, iv1, iv2):
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
		o.write("    <p><span> X = %s </span>" % ', X = '.join([ '%.1f' % elem for elem in coordinates[:,0] ]))
		o.write("    </p>")
		o.write("    <a href='.%s/X_Slices.png'>\n" % outname)
		o.write("    <img src = '.%s/X_Slices.png' width='100%%'>\n" % outname)
		o.write("    </a>\n")
		o.write("  <h1> Y Axis </h1>\n")
		o.write("    <p><centre> Y = %s</centre>" % ', Y = '.join([ '%.1f' % elem for elem in coordinates[:,1] ]))
		o.write("    </p>")
		o.write("    <a href='.%s/Y_Slices.png'>\n" % outname)
		o.write("    <img src = '.%s/Y_Slices.png' width='100%%'>\n" % outname)
		o.write("    </a>\n")
		o.write("  <h1>Z Axis </h1>\n")
		o.write("    <p>Z = %s " % ', Z = '.join([ '%.1f' % elem for elem in coordinates[:,2] ]))
		o.write("    </p>")
		o.write("    <a href='.%s/Z_Slices.png'>\n" % outname)
		o.write("    <img src = '.%s/Z_Slices.png' width='100%%'>\n" % outname)
		o.write("    </a>\n")
		o.write("    <h1> Color Bar(s) </h1>\n")
		o.write("    <ul>\n")
		o.write("        <a href='.%s/0_colorbar.png'>\n" % outname)
		o.write("        <li><img src='.%s/0_colorbar.png' width='auto' height='200'></li>\n" % outname)
		o.write("        </a>\n")
		if iv1 is not None:
			o.write("        <a href='.%s/1_colorbar.png'>\n" % outname)
			o.write("        <li><img src='.%s/1_colorbar.png' width='auto' height='200'></li>\n" % outname)
			o.write("        </a>\n")
		if iv2 is not None:
			o.write("        <a href='.%s/2_colorbar.png'>\n" % outname)
			o.write("        <li><img src='.%s/2_colorbar.png' width='auto' height='200'></li>\n" % outname)
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
