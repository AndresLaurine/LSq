from mesh import Mesh
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsmr
import importlib

m = Mesh("shell/slice.obj")
nboundary = sum(m.boundary)

model = "shell"
attr = importlib.import_module(model + ".attributes")

for c in range(m.ncorners): # lift all vertices of all horizons
    if attr.horizon_id[c]>=0:
        height = (1+attr.horizon_id[c])/37.76 # arbitrarily chosen coeff to get a visually nice result
        m.V[m.org(c)][2] = m.V[m.dst(c)][2] = height

nb_horizon = 0
for n in attr.horizon_id:
    if nb_horizon < n:
        nb_horizon = n

for dim in range(2): # solve for x first, then for y
    A = scipy.sparse.lil_matrix((nboundary + m.ncorners, m.nverts))
    b = [0] * A.shape[0]
    liste_max = [-100000 for _ in range(nb_horizon + 2)]

    for row in range(m.ncorners):
        i = m.org(row)
        j = m.dst(row)
        A[row, j] = 1
        A[row, i] = -1
        hor = attr.horizon_id[row]
        if liste_max[hor] < m.V[i][0]:
            liste_max[hor] = m.V[i][0]

    for cor in range(m.ncorners):
        i = m.org(cor)
        hor = attr.horizon_id[cor]
        if hor >= 0:
            A[cor, i] = 1
            b[cor] = liste_max[hor]

    row = cor + 1
    for (i,v) in enumerate(m.V):
        if m.on_border(i):
            A[row, i] = 100  # quadratic penalty to lock boundary vertices
            b[row] = v[dim]*100
            row += 1


    A = A.tocsr() # convert to compressed scorrse row format for faster matrix-vector muliplications
    x = lsmr(A, b)[0] # call the least squares solver
    for i in range(m.nverts): # apply the computed flattening
        m.V[i][dim] = x[i]

m.write_vtk("output1.vtk")
#print(m) # output the deformed mesh

