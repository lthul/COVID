import numpy as np
import gurobi as gp

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

nc = 3
N = np.random.randint(1000, size=nc)
m = gp.Model("matrix1")

# Create variables
x = m.addMVar(shape=3, vtype=GRB.INTEGER, name="x")

# Set objective
gp.max_()
m.setObjective(obj @ x, GRB.MAXIMIZE)

A = np.ones(nc)

# Build rhs vector
rhs = np.array([10])

# Add constraints
m.addConstr(A @ x <= rhs,  name="c")
m.addConstr(0 <= x)

# Optimize model
m.optimize()

print(x.X)
print('Obj: %g' % m.objVal)
print()