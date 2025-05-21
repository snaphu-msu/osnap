import numpy as np
from scipy.interpolate import RegularGridInterpolator
from SkyNet import *

def get_level(ds,dds,nblockx,blocksize):
    xmin = ds.domain_left_edge[0].v
    xmax = ds.domain_right_edge[0].v
    level0_dx = (xmax-xmin)/nblockx/blocksize
    level = np.log(level0_dx/dds)/np.log(2)
    
    return level

def get_starting_block(dsstart,nblockx):
    
    level0grids = []
    for grid in dsstart.index.grids:
        if grid.Parent==None: level0grids.append(grid)

    xmin = dsstart.domain_left_edge[0]
    xmax = dsstart.domain_right_edge[0]
    
    deltax = 0.1*(xmax-xmin)/nblockx

    level0_blockedge = (xmax-xmin)/nblockx #must be same for each dim
    level0 = {}
    for grid in level0grids:
        ix = int((grid.LeftEdge[0]-xmin+deltax)/level0_blockedge)
        level0[ix] = grid
        
    return level0_blockedge, level0

def cascade_to_leaf(dsstart,point_pos,level0_blockedge,level0,blocksize):
    topblock = ((point_pos-dsstart.domain_left_edge.v)/level0_blockedge.v).astype(int)
    grid = level0[topblock[0]]
    level=0
    while len(grid.Children) > 1:
        level += 1
        RE = grid.RightEdge.v
        LE = grid.LeftEdge.v
        child = (2.0*(point_pos-LE)/(RE-LE)).astype(int)
        grid = grid.Children[child[0] + 2*child[1] + 4*child[2]]

    RE = grid.RightEdge.v
    LE = grid.LeftEdge.v
    ix = int(blocksize*(point_pos[0]-LE[0])/(RE[0]-LE[0]))
    return grid,level,(ix,0,0)

def shift_data(data,CG0v,CG1v,reflect=1):
    data[0][1:] = CG1v.v[:-1,0,0]
    data[1][1:] = CG0v.v[:-1,0,0]
    data[0][0] = reflect*CG1v.v[0,0,0]
    data[1][0] = reflect*CG0v.v[0,0,0]                          
                               
def setup_CGs(grid,posindices,level,dsstart,dsend):
    #this setup is for 3D octant
    left_edge = [grid.LeftEdge[0]-grid.dds[0],grid.LeftEdge[1],grid.LeftEdge[2]]
    left_edge += posindices*grid.dds[0]
    #trick CGs to get only data in domain, will shift below and use reflecting BCs

    if (grid.LeftEdge[0]==0): left_edge[0] = grid.LeftEdge[0]
                               
    CG0 = dsstart.covering_grid(level=level,left_edge=left_edge,dims=[3,1,1])  
    CG1 = dsend.covering_grid(level=level,left_edge=left_edge,dims=[3,1,1])          

    #get back intended grid
    left_edge = [grid.LeftEdge[0]-grid.dds[0],grid.LeftEdge[1],grid.LeftEdge[2]]
    left_edge += posindices*grid.dds[0]
    
    x = np.asarray(np.linspace(left_edge[0],left_edge[0]+grid.dds[0]*3,3,endpoint=False)+grid.dds[0]*0.5)

    t = np.asarray([dsend.current_time.v.item(),dsstart.current_time.v.item()])
        
    velx = np.array([CG1['velx'].v[:,0,0],CG0['velx'].v[:,0,0]])
                     
    temp = np.array([CG1['temp'].v[:,0,0],CG0['temp'].v[:,0,0]])
    dens = np.array([CG1['dens'].v[:,0,0],CG0['dens'].v[:,0,0]])
    ye = np.array([CG1['ye  '].v[:,0,0],CG0['ye  '].v[:,0,0]])
    dye = np.array([CG1['dye '].v[:,0,0],CG0['dye '].v[:,0,0]])
#    lume = np.array([CG1['fnue'].v[:,0,0],CG0['fnue'].v[:,0,0]])
#    avee = np.array([CG1['enue'].v[:,0,0],CG0['enue'].v[:,0,0]])
#    rmse = np.array([CG1['rnue'].v[:,0,0],CG0['rnue'].v[:,0,0]])
#    luma = np.array([CG1['fnua'].v[:,0,0],CG0['fnua'].v[:,0,0]])
#    avea = np.array([CG1['enua'].v[:,0,0],CG0['enua'].v[:,0,0]])
#    rmsa = np.array([CG1['rnua'].v[:,0,0],CG0['rnua'].v[:,0,0]])
    
    #perhaps need to fix be those grids with LeftEdge=0 have CG
    #data that is shifted by 1 zone, need to reflect the first zone and shift
    if (grid.LeftEdge[0]==0):
        shift_data(velx,CG0['velx'],CG1['velx'],reflect=-1)
        shift_data(temp,CG0['temp'],CG1['temp'],reflect=1)
        shift_data(dens,CG0['dens'],CG1['dens'],reflect=1)
        shift_data(ye,CG0['ye  '],CG1['ye  '],reflect=1)
        shift_data(dye,CG0['dye '],CG1['dye '],reflect=1)
#        shift_data(lume,CG0['fnue'],CG1['fnue'],reflect=1)
#        shift_data(avee,CG0['avee'],CG1['avee'],reflect=1)
#        shift_data(rmse,CG0['rmse'],CG1['rmse'],reflect=1)
#        shift_data(luma,CG0['fnua'],CG1['fnua'],reflect=1)
#        shift_data(avea,CG0['avea'],CG1['avea'],reflect=1)
#        shift_data(rmsa,CG0['rmsa'],CG1['rmsa'],reflect=1)
    rgi = {}
    rgi['velx'] = RegularGridInterpolator((t, x), velx)
    rgi['temp'] = RegularGridInterpolator((t, x), temp)
    rgi['dens'] = RegularGridInterpolator((t, x), dens)
    rgi['ye'] = RegularGridInterpolator((t, x), ye)
    rgi['dye'] = RegularGridInterpolator((t, x), dye)
#    rgi['lume'] = RegularGridInterpolator((t, x), lume)
#    rgi['avee'] = RegularGridInterpolator((t, x), avee)
#    rgi['rmse'] = RegularGridInterpolator((t, x), rmse)
#    rgi['luma'] = RegularGridInterpolator((t, x), luma)
#    rgi['avea'] = RegularGridInterpolator((t, x), avea)
#    rgi['rmsa'] = RegularGridInterpolator((t, x), rmsa)
    
    return x,t,rgi

def evolve_back_one_file_many(dsstart,dsend,starting_points,nsub,nblockx,blocksize,NSEtemp):
    #first get level0 blocks in dsstart
    level0_blockedge, level0 = get_starting_block(dsstart,nblockx)
    
    p = 0
    points = list(range(len(starting_points)))

    for starting_point in starting_points:
        if starting_point[4] > NSEtemp: 
            points[p] = starting_point
            p += 1
            continue
        #now find target level
        grid, level, posindices = cascade_to_leaf(dsstart,starting_point[1:4],level0_blockedge,level0,blocksize)

        #now get covering grids
        x,t,rgi = setup_CGs(grid,posindices,level,dsstart,dsend)

        #check for rho,t,ye
        if starting_point[4]==0.0:
            #print(starting_point[:4],x,y,z)
            starting_point[4] = rgi['temp'](starting_point[:2]).item()
            starting_point[5] = rgi['dens'](starting_point[:2]).item()
            starting_point[6] = rgi['ye'](starting_point[:2]).item()
            starting_point[7] = rgi['dye'](starting_point[:2]).item()
#            starting_point[8] = rgi['lume'](starting_point[:2]).item()
#            starting_point[9] = rgi['avee'](starting_point[:2]).item()
#            starting_point[10] = rgi['rmse'](starting_point[:2]).item()
#            starting_point[11] = rgi['luma'](starting_point[:2]).item()
#            starting_point[12] = rgi['avea'](starting_point[:2]).item()
#            starting_point[13] = rgi['rmsa'](starting_point[:2]).item()
    
        #trace through dt step
        t0 = dsstart.current_time.v.item()
        deltat = dsend.current_time.v.item()-dsstart.current_time.v.item()
        dt = deltat/nsub
        point = starting_point[:4]
        for i in range(nsub):
            if i==0: point[0] -= 1e-8
            try:
                lvelx = rgi['velx'](point[:2]).item()
            except:
                print("outside error?",starting_point,point,x)
                print(rgi['velx'].grid,rgi['velx'].values)
                lvelx = rgi['velx'](point[:2]).item()
            point = np.add(point, (dt,dt*lvelx,0.0,0.0)) 
            if point[1]<0.0:
                point[1] = 1e-20
            if point[1]<x[0] or point[1]>x[-1]:
                grid, level,posindices = cascade_to_leaf(dsstart,point[1:],level0_blockedge,level0,blocksize)
                x,t,rgi = setup_CGs(grid,posindices,level,dsstart,dsend)
            if i==0: point[0] += 1e-8
        point[0] += 1e-8
        ltemp = rgi['temp'](point[:2]).item()
        ldens = rgi['dens'](point[:2]).item()
        lye = rgi['ye'](point[:2]).item()
        ldye = rgi['dye'](point[:2]).item()
#        llume = rgi['lume'](point[:2]).item()
#        lavee = rgi['avee'](point[:2]).item()
#        lrmse = rgi['rmse'](point[:2]).item()
#        lluma = rgi['luma'](point[:2]).item()
#        lavea = rgi['avea'](point[:2]).item()
#        lrmsa = rgi['rmsa'](point[:2]).item()
        point[0] -= 1e-8
#        try:
#            Te,etae = newton_raphson(f_avee,f_rmse,df_dT_avee,df_deta_avee,df_dT_rmse,df_deta_rmse,lavee,lrmse**2,lavee/3.15,0.0)
#        except:
#            print(lavee,lrmse)
        
#        try:
#            Ta,etaa = newton_raphson(f_avee,f_rmse,df_dT_avee,df_deta_avee,df_dT_rmse,df_deta_rmse,lavea,lrmsa**2,lavea/3.15,0.0)
#        except:
#            print(lavea,lrmsa)
        
#        points[p] = np.asarray([point[0],point[1],point[2],point[3],ltemp,ldens,lye,ldye,llume,Te,etae,lluma,Ta,etaa,starting_point[14]])
        points[p] = np.asarray([point[0],point[1],point[2],point[3],ltemp,ldens,lye,ldye,starting_point[8]])
        p += 1
    return np.asarray(points).reshape(1,len(starting_points),9)

def load_kepler_model_SWH2018_solar(filename,rows_to_header):
    
    #first get header with column names
    fs = open(filename,'r')
    for i in range(rows_to_header):
        fs.readline()
    header = fs.readline()
    fs.close()
    
    data = np.genfromtxt(filename,skip_header=rows_to_header+1)
    return header,data

def load_mesa_model_tardis(filename,rows_to_header):
    
    #first get header with column names
    fs = open(filename,'r')
    for i in range(rows_to_header):
        fs.readline()
    header = fs.readline()
    fs.close()
    
    data = np.genfromtxt(filename,skip_header=rows_to_header+1)
    return header,data

skynetA = (1, 1, 2, 3, 3, 4, 6, 6, 7, 8, 9, 7, 9, 8, 10, 11, 9, 10, 11, 12, 13,
   14, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 17, 18, 19, 17, 18, 19,
   20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23, 24, 25, 26, 22, 23,
   24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 28, 29, 30, 26, 27, 28, 29,
   30, 31, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 31, 32, 33, 34, 35,
   36, 37, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 35, 36, 37, 38, 39,
   40, 41, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44,
   45, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 43, 44, 45, 46,
   47, 48, 49, 50, 51, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
   46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 46, 47, 48, 49, 50, 51, 52,
   53, 54, 55, 56, 57, 58, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 50,
   51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 56, 57, 58,
   59, 60, 61, 62, 63, 64, 65, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
   65, 66, 67, 68, 69, 70, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
   71, 72, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
   74, 75, 76, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
   65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
   82, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 67, 68,
   69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
   86, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
   72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
   89, 90, 91, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 93, 94, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
   89, 90, 91, 92, 93, 94, 95, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
   92, 93, 94, 95, 96, 97, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 93, 94, 95, 96, 97, 98, 86, 87, 88, 89, 90, 91, 92, 93, 94,
   95, 96, 97, 98, 99, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
   98, 99, 100, 101, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
   101, 102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
   101, 102, 103, 104, 105, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
   103, 104, 105, 106, 107, 108, 109, 92, 93, 94, 95, 96, 97, 98, 99,
   100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 98, 99, 100,
   101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
   114, 115, 116, 117, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
   106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 124, 104, 105, 106, 107, 108, 109, 110, 111,
   112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
   125, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
   116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
   129, 130, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
   108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
   121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
   134, 135, 136)

skynetZ =  (0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7,
   7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11,
   11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14,
   14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16,
   16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18,
   18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
   20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
   22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23,
   24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25,
   25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
   26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28,
   28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29,
   29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
   30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32,
   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33,
   33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34,
   34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35,
   35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36,
   36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37,
   37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38,
   38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39,
   39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40,
   40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
   40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
   41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
   42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
   44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45,
   45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46,
   46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47,
   47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48,
   48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
   49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
   49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
   50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51,
   51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
   51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
   52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53,
   53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53,
   53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
   54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54)

element_list = ['h',   'he',  'li',  'be',  'b',   'c',   'n',
     'o',   'f',   'ne',  'na',  'mg',  'al',  'si',  'p',
     's',   'cl',  'ar',  'k',   'ca',  'sc',  'ti',  'v',
     'cr',  'mn',  'fe',  'co',  'ni',  'cu',  'zn',  'ga',
     'ge',  'as',  'se',  'br',  'kr',  'rb',  'sr',  'y',
     'zr',  'nb',  'mo',  'tc',  'ru',  'rh',  'pd',  'ag',
     'cd',  'in',  'sn',  'sb',  'te',  'i',   'xe',  'cs',
     'ba',  'la',  'ce',  'pr',  'nd',  'pm',  'sm',  'eu',
     'gd',  'tb',  'dy',  'ho',  'er',  'tm',  'yb',  'lu',
     'hf',  'ta',   'w',  're',  'os',  'ir',  'pt',  'au',
     'hg',  'tl',  'pb',  'bi',  'po',  'at',  'rn',  'fr',
     'ra',  'ac',  'th',  'pa',  'u',   'np',  'pu',  'am',
     'cm',  'bk',  'cf',  'es',  'fm',  'md',  'no',  'lr',
     'rf',  'db',  'sg',  'bh',  'hs',  'mt',  'ds',  'rg',
     'cn',  'ut',  'fl',  'up',  'lv',  'us',  'uo']

def skynet_get_initY(prog_header,prog_radius,prog_data,radius,skynetA,skynetZ,prog_compstart):
    initY = np.zeros(len(skynetA))
    index = (np.where(prog_radius-radius>0))[0][0]
    
    data1 = prog_data[index-1]
    data2 = prog_data[index]
    
    frac1 = (prog_radius[index]-radius)/(prog_radius[index]-prog_radius[index-1])
    frac2 = 1.0-frac1
    
    zonedata = frac1*prog_data[index-1] + frac2*prog_data[index]

    keplercomps={}
    for i in range(prog_compstart-1,31):
        #print(prog_header.split())
        spd = prog_header.split()
        #print(i,spd[i+1],prog_header[7+(i-1)*25:7+(i)*25])
        #iso = spd[i+1].strip()
        iso = prog_header[7+(i-1)*25:7+(i)*25].strip()
        #print(iso)
        keplercomps[iso] = zonedata[i+1]
        #print(zonedata[i+1])

    for i in range(len(skynetA)):
        if skynetZ[i]==0:
            iso = 'neutrons'
        else:
            element = element_list[skynetZ[i]-1][0].upper()+element_list[skynetZ[i]-1][1:] 
            iso = element+str(skynetA[i])

        try: 
            initY[i] = keplercomps[iso]
            #print(iso,keplercomps[iso],radius)
        except:
            #print("Element not in kepler",iso)
            initY[i] = 0.0

    #print(sum(initY))
            
       
    initY = initY/sum(initY)
    initY /= skynetA

    return initY

def skynet_get_initY_mesa(header,prog_radius,prog_data,radius,skynetA,skynetZ,prog_compstart):
    initY = np.zeros(len(skynetA))

    print(prog_radius)
    index = (np.where(prog_radius-radius<0))[0][0]
    
    data1 = prog_data[index-1] #mesa, so this is data at the larger radius
    data2 = prog_data[index] #data at the smaller radius
    
    frac2 = (prog_radius[index-1]-radius)/(prog_radius[index-1]-prog_radius[index])
    frac1 = 1.0-frac2
    
    zonedata = frac1*prog_data[index-1] + frac2*prog_data[index]

    mesacomps={}
    for i in range(prog_compstart,prog_compstart+22):
        iso = header[i].strip()
        mesacomps[iso] = zonedata[i]
        #print(iso,zonedata[i])

    for i in range(len(skynetA)):
        if skynetZ[i]==0:
            iso = 'neut'
        else:
            element = element_list[skynetZ[i]-1][0].upper()+element_list[skynetZ[i]-1][1:] 
            iso = (element+str(skynetA[i])).lower()

        try: 
            initY[i] = mesacomps[iso]
            #print(iso,mesacomps[iso],radius)
        except:
            #print("Element not in mesa",iso)
            initY[i] = 0.0

    #print(sum(initY))
            
       
    initY = initY/sum(initY)
    initY /= skynetA

    return initY

def run_skynet(do_inv, do_screen,tfinal,trajectory_data,outfile,do_NSE=True,initY=None):

    pref = outfile

    with open(SkyNetRoot + "/examples/code_tests/X-ray_burst/sunet") as f:
        nuclides = [l.strip() for l in f.readlines()]

    nuclib = NuclideLibrary.CreateFromWinv(SkyNetRoot + "/examples/code_tests/winvne_v2.0.dat", nuclides)

    opts = NetworkOptions()
    opts.ConvergenceCriterion = NetworkConvergenceCriterion.Mass
    opts.MassDeviationThreshold = 1.0E-10
    opts.IsSelfHeating = False
    opts.EnableScreening = do_screen
    opts.DisableStdoutOutput = True
    opts.MaxDt=0.001

    helm = HelmholtzEOS(SkyNetRoot + "/data/helm_table.dat")

    strongReactionLibrary = REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/reaclib",
        ReactionType.Strong, do_inv, LeptonMode.TreatAllAsDecayExceptLabelEC,
        "Strong reactions", nuclib, opts, True, True)
    weakReactionLibrary = REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/reaclib",
        ReactionType.Weak, False, LeptonMode.TreatAllAsDecayExceptLabelEC,
        "Weak reactions", nuclib, opts, True, True)
    symmetricFission = REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/nfis",
        ReactionType.Strong, False, LeptonMode.TreatAllAsDecayExceptLabelEC,
        "Symmetric neutron induced fission with 0 neutrons emitted",
        nuclib, opts, True, True)
    spontaneousFission = REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/sfis",
        ReactionType.Strong, False, LeptonMode.TreatAllAsDecayExceptLabelEC,
        "Spontaneous fission", nuclib, opts, True, True)

    reactionLibraries = [strongReactionLibrary, weakReactionLibrary,
        symmetricFission, spontaneousFission]

    screen = SkyNetScreening(nuclib)
    net = ReactionNetwork(nuclib, reactionLibraries, helm, screen, opts)

    dat = trajectory_data #np.loadtxt(file)
    density_vs_time = PiecewiseLinearFunction(dat[:,0], dat[:,2], True)
    temperature_vs_time = PiecewiseLinearFunction(dat[:,0], dat[:,1], True)

    Ye0 = dat[0,3]
    
    Temp0 = dat[0,1]
    t0 = dat[0,0] + 1.0e-20

    initdt = 0.1*(dat[1,0]-dat[0,0])
    if do_NSE:
        output = net.EvolveFromNSE(t0, tfinal, temperature_vs_time, density_vs_time,Ye0,pref)
    else:
        output = net.Evolve(initY, t0, tfinal, temperature_vs_time, density_vs_time,pref)

    return output

def combine_ID_with_PE(ID,PE,radius_to_join,vars):
    
    startIDatindex,endPEatindex = get_cutoffs(ID,PE,radius_to_join)

    OD = {}
    for var in vars:
        outdata = ID[var][startIDatindex:] 
        indata = PE[var][:endPEatindex+1]
        OD[var] = np.concatenate((indata,outdata))

    return OD

def print_flash_ID_file(data,filename,vars):
    fs = open(filename,'w')
    fs.write("#combined output from flash and ID to make bigger, post explosion ID\n")
    fs.write("number of variables = "+str(len(data)-1)+"\n")
    for var in vars:
        if var!='rad': fs.write(var+"\n")
    for i in range(len(OD['rad'])):
        for var in vars:
            fs.write(str(OD[var][i])+' ')
        fs.write('\n')
    fs.close()
