# TMI_Viewer: Vertex/Voxel Rendering for TMI files and other neuroimaging formats

![Splash type schematic](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/doc/tmi_viewer_multimodal.png "Schematic")

The tmi_viewer package contains two programs:
### A) tmi_viewer

![tmi_viewer example](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/doc/TMI_viewer_example.png)

Displays multimodal neuroimaging data (surfaces and volumes) in the same space from a TMI file (although TMI files are not required).

Features:

- Multiple surfaces with vertex painting can be easily viewed with voxel images.
- Voxel images can be viewed as: (1) surfaces using a marching cube algorithm, (2) voxel contour, (3) and voxel scalar field.
- The default settings are optimized for viewing neuroimages. 
- tmi_viewer is highly optimized for speed. 
- Many autothresholding algorithms available including: Otsu et al., Li et al., Yen et al., and Z threholding.
- Extremely fast algorithm for applying Lapacian or Taubin (low-pass) smooth. e.g., ~1000 passes takes around one minute.
- Easy export of background transparent images for creating figures.
- Many new look-up tables (LUTs) that are specifically designed for visualising neuroimaging statistics (as well as the LUTs included with matplotlib).

![tmi_viewer LUTs](https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/doc/TMI_VIEWER_LUTS.png)

### B) tm_slices

Outputs a web-page with whole brain slices from voxel-based neuroimages in native coordinates with optional overlaps.

Features:

- Great for making figures: Creates a web-page displaying overlapping any number of voxel-based images, and they can be at any resolution.
- Many autothresholding algorithms available including: Otsu et al., Li et al., Yen et al., and Z threholding.
- Import images, binarize them at any threshold, and paint the image outline.
- Specify number of slices, size of slices, transparency, etc.

## Installation

### With sudo permissions

##### Requirements:

* [TFCE_mediation](https://github.com/trislett/TFCE_mediation)
* [VTK](http://www.vtk.org/download/)

For Ubuntu:
```sudo apt-get install vtk6 python-vtk python-qt4```

For OSX:
```brew install vtk```

##### Installation:

Using PIP (Recommended):
```sudo -H pip install -U tmi_viewer```

From source:
```sudo python setup.py install```
 - Additional requirement: [mayavi](http://docs.enthought.com/mayavi/mayavi/)
 
### Using a Python Virtual Environment
 
_Note: this example uses python 3.5_
 
**Create and source virtual environment**

```
virtualenv -p python3.5 python3env
source python3env/bin/activate
```

**Install TFCE_mediation**

```
pip install -U tfce-mediation

```

**Download and install SIP and PyQT4**

a. Download and unzip SIP and PyQT4

* SIP download [link](https://www.riverbankcomputing.com/software/sip/download)
* QT4 download [link](https://www.riverbankcomputing.com/software/pyqt/download)

b. Install SIP (not version may be different)

```
cd sip-4.19.7
python configure.py
make
make install
```

c. Install PyQT4

```
cd ../PyQt-x11-gpl-4.12.1
python configure-ng.py
```

Accepted the user agreement then run:

```
make
make install
```

**pip install vtk**

```
pip install vtk
```

**pip install tmi_viewer**

```
pip install -U tmi_viewer
```

***

These programs relies on Mayavi, and setting can changed using the Mayavi interactive session. If you use them please cite: 

Ramachandran, P. and Varoquaux, G., _Mayavi. 3D Visualization of Scientific Data._ IEEE Computing in Science & Engineering, 13 (2), pp. 40-51 (2011).


***
