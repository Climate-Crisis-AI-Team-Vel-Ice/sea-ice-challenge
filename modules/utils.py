import csv 
import numpy as np 
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mp
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import from_levels_and_colors
import math 
import calendar


'''
reads lat and lon csv files and creates a 3D array 
'''
def generate_ease_grid(lat_csv, lon_csv):
    lat = genfromtxt(lat_csv, delimiter=',')   #latitude
    lon = genfromtxt(lon_csv, delimiter=',')   #longitude 

    shape = (lat.shape[0], lat.shape[1], 2)
    grid = np.zeros(shape, dtype=float)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i][j][0] = lat[i][j]
            grid[i][j][1] = lon[i][j]

    return grid

'''
reads the main dataset into a dataframe and does the following manipulation
'''
def read_dataset(path):
    df = pd.read_csv(path) 
    '''
    place holder for basic dataframe manipulation
    '''
    data = df

    return data

'''
the function inputs x, y coordinates that are already projected on the map and visualize them.
it also shows the EASE grid
in the following:
-> pos_x is the list of x positions on the map
-> pos_y is the list of y positions on the map
-> fig is the pyplot figure object created in the main script
-> lat_0 and lon_0 is the center of map
-> instance of basemap returns a map with a requested projection.
-> projection can be (not limited to):
     Cylindrical: 'cyl'
     Conic: 'lcc'
     Perspective: 'ortho'
     Pseudo-cylindrical: 'moll'

-> map(lat,lon) returns an x,y position on the map
'''
def draw_map(lat, lon, size, color, grid, fig, projection='cyl', 
            center=(90,0), show_grid=True, grid_res=10, 
            width=4e6, height=3e6, year=2000, month=2, doy=[], show_text=True):

    # set up the desired map projection and background color
    if projection == 'cyl':
        m = Basemap(projection='cyl', resolution=None,
                    llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180)
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'ortho':
        m = Basemap(projection='ortho', resolution=None,
                    lat_0=center[0], lon_0=center[1])
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'aeqd':
        m = Basemap(width=width,height=width,projection='aeqd',
                    lat_0=center[0], lon_0=center[1])
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'lcc':
        m = Basemap(projection='lcc', resolution=None,
                    lon_0=0, lat_0=90, lat_1=45, lat_2=55,
                    width=width, height=height)
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'stere':
        m = Basemap(projection='stere', resolution=None,
                    lon_0=0, lat_0=90, 
                    width=width, height=height)
          
        # m.etopo(scale=0.5, alpha=1.0)    
        m.bluemarble(scale=0.5)

    else:
        print('projection not supported')
   
    # show the ease grid if requested
    if show_grid:
        # create an array to hold all x and y positions
        x_coords = []
        y_coords = []

        # Map (long, lat) to (x, y) for plotting
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if (i%grid_res == 0 and j%grid_res == 0) or (i == grid.shape[0] or j == grid.shape[1]):
                    x, y = m(grid[i][j][1], grid[i][j][0])
                    x_coords.append(x)
                    y_coords.append(y)

        plt.plot(x_coords, y_coords, 'or', markersize=1)

    # show the main data on the map
    Gradient = plt.get_cmap('Reds')   # create color gradient based on ice thickness
    marker_color = Gradient(color)

    # plot the buoy data on the map
    if lat != [] and lon != []:
        x, y = m(lon, lat)
        plt.scatter(x, y, c=marker_color,  marker='o', s=200)
        #plt.plot(x, y, color=marker_color, markersize=10)
        #plt.plot(x,y, color=color_list, marker='o', linewidth=2, markersize=10)
        #plt.text(40000, 40000, str(year)+', '+ calendar.month_abbr[month], fontsize=24,color='red')
        if show_text:
            plt.text(40000, 40000, str(year)+', '+ str(doy), fontsize=24,color='red')
        
    if doy != []:
        plt.savefig('pictures/{}/{}.png'.format(year, month))
        plt.savefig('pictures/all/{}-{}.png'.format(year, month))
    else:
        plt.savefig('pictures/{}/{}.png'.format(year, doy))
        plt.savefig('pictures/all/{}-{}.png'.format(year, doy))        

'''
special visualization helper for plots without month
'''
def visualize(lat, lon, grid, fig, projection='cyl', 
            center=(90,0), show_grid=True, grid_res=10, 
            width=4e6, height=3e6, year=2000, show_text=True, thick=[]):

    # set up the desired map projection and background color
    if projection == 'cyl':
        m = Basemap(projection='cyl', resolution=None,
                    llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180)
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'ortho':
        m = Basemap(projection='ortho', resolution=None,
                    lat_0=center[0], lon_0=center[1])
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'aeqd':
        m = Basemap(width=width,height=width,projection='aeqd',
                    lat_0=center[0], lon_0=center[1])
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'lcc':
        m = Basemap(projection='lcc', resolution=None,
                    lon_0=0, lat_0=90, lat_1=45, lat_2=55,
                    width=width, height=height)
        m.etopo(scale=0.5, alpha=0.5)

    elif projection == 'stere':
        m = Basemap(projection='stere', resolution=None,
                    lon_0=0, lat_0=90, 
                    width=width, height=height)
          
        # m.etopo(scale=0.5, alpha=1.0)    
        m.bluemarble(scale=0.5)

    else:
        print('projection not supported')
   
    # show the ease grid if requested
    if show_grid:
        # create an array to hold all x and y positions
        x_coords = []
        y_coords = []

        # Map (long, lat) to (x, y) for plotting
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if (i%grid_res == 0 and j%grid_res == 0) or (i == grid.shape[0] or j == grid.shape[1]):
                    x, y = m(grid[i][j][1], grid[i][j][0])
                    x_coords.append(x)
                    y_coords.append(y)

        plt.plot(x_coords, y_coords, 'or', markersize=1)

    Gradient = plt.get_cmap('Reds')   # create color gradient based on ice thickness
    marker_color = Gradient(thick)
    # plot the buoy data on the map
    if lat != [] and lon != []:
        x, y = m(lon, lat)
        plt.scatter(x, y, c=marker_color,  marker='o', s=50)

        if show_text:
            plt.text(80000, 80000, str(year), fontsize=36, color='red')
        
        plt.savefig('pictures/vis/{}.png'.format(year))
     
'''
interpolates the latitude and longitude of a buoy from its cartesian coordinates (x_EASE, y_EASE) on the EASE grid
'''
def interpolate_coordinate(x, y, grid):
    i = int(x)
    j = int(y)

    lat_0 = grid[i][j][0]
    lat_1 = grid[i+1][j][0]
    lon_0 = grid[i][j][1]
    lon_1 = grid[i][j+1][1]    
    lat_interpolated = lat_0 + (lat_1 - lat_0)*(x - i)
    lon_interpolated = lon_0 + (lon_1 - lon_0)*(y - j)  

    return lat_interpolated, lon_interpolated

'''
converts u and v components of velocity to magnitude and direction
'''
def caonvert_vel_vector(u,v):
    mag = math.sqrt(u**2 + v**2)
    dir = math.atan2(v, u)
    # if dir < 0:
    #     dir += 2*3.1416
    
    return mag, dir

def relative_buoy_wind_vector(wu, wv, bu, bv):
    # buoy with respect to wind
    sub = [bu-wu, bv-wv]
    wmag = math.sqrt(wu**2 + wv**2)
    bmag = math.sqrt(bu**2 + bv**2)
    # rel_mag = math.sqrt(sub[0]**2 + sub[1]**2)
    rel_mag = bmag - wmag

    # angle of buoy velocity vector with +x
    dir_buoy = math.atan2(bv, bu)

    # angle of wind velocity vector with +x 
    dir_wind = math.atan2(wv, wu)

    # direction of buoy w/r to wind
    rel_dir = dir_buoy - dir_wind

    if rel_dir > math.pi:
        rel_dir = 2*math.pi - rel_dir 

    elif rel_dir < -math.pi:
        rel_dir = - (2*math.pi + rel_dir)
    # if wmag != 0 and bmag != 0:
    #     cos_theta = (wu*bu+wv*bv) / (wmag*bmag)
    # else:
    #     return rel_mag, 0

    # rel_dir = math.acos(cos_theta) 

    return rel_mag, rel_dir