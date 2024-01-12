from mesh import Mesh
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsmr
import importlib


model = "shell"
m = Mesh(model+"/slice.obj")
nboundary = sum(m.boundary)


attr = importlib.import_module(model + ".attributes")
"""
for c in range(m.ncorners): # lift all vertices of all horizons
    if attr.horizon_id[c]>=0:
        height = (1+attr.horizon_id[c])/37.76 # arbitrarily chosen coeff to get a visually nice result
        m.V[m.org(c)][2] = m.V[m.dst(c)][2] = height
"""

nb_horizon = 0
for n in attr.horizon_id:
    if nb_horizon < n:
        nb_horizon = n
        
liste_max = [-100000 for _ in range(nb_horizon + 2)]
liste_bord = [[100,100],[-100,-100]]

for row in range(m.ncorners):

    ### creation of a list of all the y_max for each horizon
    i = m.org(row)
    hor_id = attr.horizon_id[row]
    if liste_max[hor_id] < m.V[i][1]:
        liste_max[hor_id] = m.V[i][1]
    ###

    ### creation of a list having the form [[x_min, y_min], [x_max,y_max]] to have the positions of the borders
    if m.on_border(i):
        for k in range(2):
            if m.V[i][k] < liste_bord[0][k]:
                liste_bord[0][k] = m.V[i][k]
            if m.V[i][k] > liste_bord[1][k]:
                liste_bord[1][k] = m.V[i][k]
    ###
                
for dim in range(2): # solve for x first, then for y
    A = scipy.sparse.lil_matrix((nboundary + m.ncorners, m.nverts))
    b = [0] * A.shape[0]

    ### lock the vertice 2 to 2 to minimize le deplacements
    for row in range(m.ncorners):
        i = m.org(row)
        j = m.dst(row)
        A[row, j] = 1
        A[row, i] = -1
        b[row] = 0.1
    ###
        
    ### horizontalization of the horizons
    if dim == 1:
        for cor in range(m.ncorners):
            i = m.org(cor)
            hor_id = attr.horizon_id[cor]
            if hor_id >= 0:
                A[cor, i] = 10
                b[cor] = 10*liste_max[hor_id]
        row = cor + 1
    ###

    ### verticalization of the faults 
        # attention : les failles sont concidérées comme des bords !    
    if dim == 0:
        for cor in range(m.ncorners):
            i = m.org(cor)
            if attr.is_fault:
                A[cor, i] = 10
                b[cor] = 10 * 


    # lock the boundaries        
    list_vertex = []

    for co in range(m.ncorners):
        i = m.org(co)
        if (m.on_border(i) and attr.horizon_id[co] == -1 and (i not in list_vertex)):
            if ((m.V[i][dim] <= liste_bord[0][dim] + 0.05 and m.V[i][dim] >= liste_bord[0][dim] - 0.05) or (m.V[i][dim] <= liste_bord[1][dim] + 0.05 and m.V[i][dim] >= liste_bord[1][dim] - 0.05 )):
                list_vertex.append(i)
                A[row, i] = 1 * 10 # quadratic penalty to lock boundary vertices
                b[row] = m.V[i][dim] * 10
                row += 1

    A = A.tocsr() # convert to compressed scorrse row format for faster matrix-vector muliplications
    x = lsmr(A, b)[0] # call the least squares solver
    for i in range(m.nverts): # apply the computed flattening
        m.V[i][dim] = x[i]

m.write_vtk(model + "_horizontal_bords_specifique.vtk")
#print(m) # output the deformed mesh

