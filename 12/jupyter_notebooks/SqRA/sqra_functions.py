import sys

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import clippedVoronoi


from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import Delaunay, delaunay_plot_2d

import itertools
from scipy.sparse.linalg import expm, expm_multiply
from shapely.geometry import Polygon, Point

def generate_random_points(Npoints, Ncells, xlim, ylim, potential, filter_potential):
    
    points = np.random.rand(Npoints, 2)
    points[:, 0] = xlim[0] + (xlim[1] - xlim[0]) * points[:, 0]
    points[:, 1] = ylim[0] + (ylim[1] - ylim[0]) * points[:, 1]
    

    # Filter points such that V(x,y) < 2
    points2 = []

    for n in range(Npoints):
        if potential(points[n,0], points[n,1]) < filter_potential:
            points2.append([points[n,0], points[n,1]])
      
    points2 = kmeans(points2, Ncells, iter=50)
    points2 = points2[0]
    
    
    return np.array(points2)

def clip_voronoi(vor, boundary_polygon):
    clipped_regions = []
    for i, region in enumerate(vor.regions):
        if not -1 in region and region:
            polygon_coords = [vor.vertices[i] for i in region]
            poly = Polygon(polygon_coords)
            intersection = poly.intersection(boundary_polygon)
            if intersection.is_empty:
                continue
            if intersection.geom_type == 'Polygon':
                clipped_regions.append(list(intersection.exterior.coords))
            else:
                for geom in intersection.geoms:
                    clipped_regions.append(list(geom.exterior.coords))
    return clipped_regions

def find_common_edge(poly1, poly2):
    # Convert input lists to Shapely polygons
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)

    # Check if the polygons intersect
    if polygon1.intersects(polygon2):
        # Check if any edge of poly1 is common with poly2
        for edge1_start, edge1_end in zip(polygon1.exterior.coords[:-1], polygon1.exterior.coords[1:]):
            for edge2_start, edge2_end in zip(polygon2.exterior.coords[:-1], polygon2.exterior.coords[1:]):
                if (edge1_start == edge2_start and edge1_end == edge2_end) or \
                   (edge1_start == edge2_end and edge1_end == edge2_start):
                    return edge1_start, edge1_end  # Return the common edge

    return None  # No common edge found


def generate_random_voronoi(Ncells, xmin, xmax, ymin, ymax):
    # Number of initial points uniformly distributed
    Npoints  = 10000
    xpoints  = np.random.uniform(xmin, xmax, Npoints)
    ypoints  = np.random.uniform(ymin, ymax, Npoints)
    points   = np.array((xpoints, ypoints))
    
    cells = kmeans(points.T,Ncells,iter=50)
    cells = cells[0]
    cc_x = cells[:,0]
    cc_y = cells[:,1]

    vor = clippedVoronoi.voronoi(cells, (xmin, xmax, ymin, ymax))

    xcenters = vor.filtered_points[:,0]
    ycenters = vor.filtered_points[:,1]
    
    return xcenters, ycenters, vor

def generate_grid_voronoi(Ncells, xmin, xmax, ymin, ymax):
    xbins  = int(np.sqrt(Ncells))
    ybins  = int(np.sqrt(Ncells))

    xedges = np.linspace(xmin, xmax, xbins + 1)  # array with x edges
    dx     = xedges[1] - xedges[0]
    x      = xedges[:-1] + (dx / 2)                # array with x centers
    xbins  = xbins - 1

    yedges = np.linspace(ymin, ymax, ybins + 1)  # array with y edges
    dy     = yedges[1] - yedges[0]
    y      = yedges[:-1] + (dy / 2)                # array with y centers
    ybins  = ybins - 1

    Nbins  = xbins*ybins                      # number of bins

    grid = np.meshgrid(x,y)

    # Grid contains 2 matrices xbins x ybins with the x and y coordinates
    # This transforms the two matrices in two vectors (x,y)

    cells_grid = np.array([grid[0].flatten('F'), grid[1].flatten('F')]).T
    vor_grid = clippedVoronoi.voronoi(cells_grid, (xmin, xmax, ymin, ymax))

    xcenters_grid = vor_grid.filtered_points[:,0]
    ycenters_grid = vor_grid.filtered_points[:,1]
    
    return xcenters_grid, ycenters_grid, dx, dy, vor_grid

def adjacency_random_voronoi(Ncells, vor):
    # Volumes    
    Vol      = np.zeros(Ncells)

    # Adjacency matrix
    A      = scipy.sparse.lil_matrix((Ncells, Ncells))
   
    # Intersecting areas
    S      = scipy.sparse.lil_matrix((Ncells, Ncells))

    # Distances between neighboring points 
    h      = scipy.sparse.lil_matrix((Ncells, Ncells))

    # Note that len(vor.filtered_regions) = Ncells
    # region = Voronoi cell
    for i,region in enumerate(vor.filtered_regions):

        vertices = vor.vertices[region + [region[0]], :]
        Vol[i] = ConvexHull(vertices).volume

        for j in range(i+1, Ncells):

            #ri and rj contain the vertices of the region omega_i and omega_j
            ri = vor.filtered_regions[i]
            rj = vor.filtered_regions[j]
            int_sur = np.intersect1d(ri, rj)

            # if two cells share vertices, 
            if int_sur.size != 0:

                # then they are adjacent:
                A[i,j] = 1
                A[j,i] = 1

                # Coordinates of the cells centers
                v0_x       = vor.filtered_points[i,0]
                v0_y       = vor.filtered_points[i,1]
                v1_x       = vor.filtered_points[j,0]
                v1_y       = vor.filtered_points[j,1]

                # Distance between the centers
                distance = np.sqrt( (v1_x - v0_x)**2 + (v1_y - v0_y)**2 )
                h[i,j]   = distance
                h[j,i]   = distance

                # Coordinates of the intersecting vertices
                int_sur0_x = vor.vertices[int_sur[0],0]
                int_sur0_y = vor.vertices[int_sur[0],1]
                int_sur1_x = vor.vertices[int_sur[1],0]
                int_sur1_y = vor.vertices[int_sur[1],1]

                # Intersecting area (length of the common edge)
                area = np.sqrt( (int_sur1_x - int_sur0_x)**2 + (int_sur1_y - int_sur0_y)**2 )
                S[i,j] = area
                S[j,i] = area
    return A, Vol, h, S

## Repeat for the grid 
def adjacency_matrix_grid(nbins, nd, periodic=False):
    v = np.zeros(nbins)
    v[1] = 1
    
    if periodic:
        v[-1] = 1
        A0 = scipy.sparse.csc_matrix(scipy.linalg.circulant(v)) #.toarray()
    else:
        A0 = scipy.sparse.csc_matrix(scipy.linalg.toeplitz(v)) #.toarray()
    
    A = A0
    I2 = scipy.sparse.eye(nbins)  #np.eye(nbins)
    for _ in range(1, nd):
        I1 = scipy.sparse.eye(*A.shape) #np.eye(*A.shape)
        A =  scipy.sparse.kron(A0, I1) + scipy.sparse.kron(I2, A)
    return A

def sqra_random_voronoi(Ncells, xcenters, ycenters, beta, D, V, A, Vol, h, S):
# Boltzmann distribution
    pi   = np.exp(-beta * V(xcenters, ycenters))
    sqra = np.sqrt(pi)

    Q = scipy.sparse.lil_matrix((Ncells, Ncells))

    for i in range(Ncells):
        for j in range(Ncells):
            if A[i,j] == 1:
                Q[i,j] = D * S[i,j] / Vol[i] / h[i,j] * sqra[j] / sqra[i]

    # The diagonal is equal to minus the row-sum
    Q    = Q + scipy.sparse.spdiags( - Q.sum(axis=1).T, 0, Ncells, Ncells)
 
    return Q

def sqra_grid_voronoi(Ncells, xcenters_grid, ycenters_grid, dx, dy, beta, D, V, A_grid):
   
    # Boltzmann distribution
    pi_grid   = np.exp(-beta * V(xcenters_grid, ycenters_grid))
    sqra_grid = np.sqrt(pi_grid)

    Q_grid = scipy.sparse.lil_matrix((Ncells, Ncells))

    for i in range(Ncells):
        for j in range(Ncells):
            if A_grid[i,j] == 1:
                Q_grid[i,j] = D / dx / dy * sqra_grid[j] / sqra_grid[i]

    # The diagonal is equal to minus the row-sum
    Q_grid    = Q_grid + scipy.sparse.spdiags( - Q_grid.sum(axis=1).T, 0, Ncells, Ncells)
    return Q_grid
