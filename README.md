# Measurements-codes
Python scripts for analysing experimental data from droplet images (angles, volumes, etc.)

Several Python scripts for analysing experimental drop data, including:
- calculation of the contact angle (using several methods)
- volume measurement
- extraction of characteristic points
- visualisation of curves and scatter plots

## Scripts

# points_depart.py
Detection of the right, left and top points of the drop on an image 

# Angle_PCA.py
Calculation of the angle using principal component analysis (PCA) 
For this function, you must also import the points_depart function.

 # methodemaxder2.py 
Calculation of the angle using the maximum derivative method

 # angle(V).py
 Calculation of the angle based on volume 
 
 # code_plots.py
 Plotting of theta(B) plots using the PCA or max der method (two different scripts commented at the end of the code)

 # nuagedepoints.py 
 Generation of a point cloud with the data (theta, B, V)
 
# nuagedepointspline.py 
 Adjustment of points via 3D spline

 # ronseiweig.py 
 To plot the Rg/2a curves, with data obtained manually on ImageJ

 # volume_integration.py
 Calculation of volume by integration (drop profile) 
