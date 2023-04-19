import sys
import itertools as it
import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=True)
#np.set_printoptions(threshold=sys.maxsize)
import sympy as sp
import matplotlib.pyplot as plt
import inspect

###Triangulation
def get_dots(dots_per_axis, x_range, y_range): #Calculates matrix of dots where each dot is [x, y]
    dots = [[(x, y) \
            for x in np.linspace(*x_range, dots_per_axis)] \
            for y in np.linspace(*y_range, dots_per_axis)]
    return np.array(dots)

def get_dots_map(dots): #Generates mapping from dot number to dot
    dots_vector = dots.reshape((-1, dots.shape[-1]))
    return dict(enumerate(dots_vector))

def get_triangles(dots): #Generates vector of triangles where each triangle is vector of dots
    triangles = []
    for yi in range(len(dots) - 1):
        for xi in range(len(dots[yi]) - 1):
            if (xi, yi) in [(0, 0), (len(dots[xi])-2, len(dots[xi])-2)]:
                triangles.append([dots[yi, xi]  , dots[yi+1, xi+1], dots[yi+1, xi  ]])
                triangles.append([dots[yi, xi]  , dots[yi, xi+1]  , dots[yi+1, xi+1]])
            else:
                triangles.append([dots[yi, xi]  , dots[yi, xi+1]  , dots[yi+1, xi]])
                triangles.append([dots[yi, xi+1], dots[yi+1, xi+1], dots[yi+1, xi]])
    return np.array(triangles)

def get_triangles_map(triangles): #Generages mapping from triangle number to triangle
    return dict(enumerate(triangles)) 

def inner_outer_dot_nums(dots, dots_map, dots_map_reverse): #Returns numbers of dots separated on inner and outer
    inner_dot_nums = [dots_map_reverse[tuple(dot)] for dot in dots[1:-1, 1:-1].reshape(-1, dots.shape[-1])]
    outer_dot_nums = list(set(range(dots.shape[0] * dots.shape[1])) - set(inner_dot_nums))
    return inner_dot_nums, outer_dot_nums

def extract_boundary_dots(triangle, x_range, y_range):
    return np.array([dot for dot in triangle if dot[0] in x_range or dot[1] in y_range])

def is_boundary_el(boundary_dots): 
    return 2 == len(boundary_dots)

def which_boundary(boundary_dots, x_range, y_range):
    bd = boundary_dots
    if   bd[0,0] == bd[1,0] == x_range[0]: return 3
    elif bd[0,0] == bd[1,0] == x_range[1]: return 1
    elif bd[0,1] == bd[1,1] == y_range[0]: return 0
    elif bd[0,1] == bd[1,1] == y_range[1]: return 2
    else: raise Exception("ConditionException")

def is_in_triangle(point, triangle):
    i, j, m = triangle
#    input(f"is_in_triangle ({point}, \n{triangle}\n)")
#    input(f"S(\n{triangle}\n) : {S(triangle)}")
#    input(f"S(\n{np.c_[point, j    , m    ].T}\n) : {S(np.c_[point, j    , m    ].T)}")
#    input(f"S(\n{np.c_[i    , point, m    ].T}\n) : {S(np.c_[i    , point, m    ].T)}")
#    input(f"S(\n{np.c_[i    , j    , point].T}\n) : {S(np.c_[i    , j    , point].T)}")
#    input(f"S(t) == S(i) + S(j) + S(m) : {S(triangle) == S(np.c_[point, j, m].T) + S(np.c_[i, point, m].T) + S(np.c_[i, j, point].T)}")
    return round(S(triangle), 8) == round(S(np.c_[point, j, m].T) + S(np.c_[i, point, m].T) + S(np.c_[i, j, point].T), 8)

def in_which_triangle(point, triangles_map):
#    print(f"in_which_triangle({point}, triangles)")
    for i, t in triangles_map.items():
        if is_in_triangle(point, t):
            return i

###FEM
def S(triangle): #Calculates area of triangle (eg. triangle=np.array([[0, 0],[1, 0],[0, 1]]) ==> 0.5)
    matrix = np.hstack((np.ones((3, 1)), triangle))
    return np.abs(.5*np.linalg.det(matrix))

def get_a_b_c(triangle): #Calculates a_i, a_j, a_m, b_i, b_j, b_m, c_i, c_j, c_m coeficients for triangle
    t = triangle
    i, j, m = range(3)
    x, y = range(2)
    return np.array([
        [t[j, x]*t[m, y] - t[m, x]*t[j, y], t[j, y] - t[m, y], t[m, x] - t[j, x]],
        [t[m, x]*t[i, y] - t[i, x]*t[m, y], t[m, y] - t[i, y], t[i, x] - t[m, x]],
        [t[i, x]*t[j, y] - t[j, x]*t[i, y], t[i, y] - t[j, y], t[j, x] - t[i, x]],
        ])

def K(triangle, delta, a_11, a_22): #Calculates K_e matrix
    _, b, c = get_a_b_c(triangle).T
    return (1/(2*delta)) * (a_11*b[:,None]@b[None,:]+a_22*c[:,None]@c[None,:])

def M(d, delta): #Calculates M_e matrix (btw. np.eye(3) + 1 returns np.array([[2,1,1],[1,2,1],[1,1,2]]))
    return (d*delta/24)*(np.eye(3)+1)

def Q(Me, d, fe): #Calculates Q_e matrix
    return (Me/d)@fe

def R(sigma, beta, Gamma):
    return sigma*Gamma/(beta*6)*(np.eye(2) + 1)

def P(psi, beta, Gamma):
    return psi*Gamma/(beta*2)*np.ones((2, 1))

def varphi(delta, a, b, c):
    return lambda x1, x2: 1/delta*(a + b*x1 + c*x2)

###Norms
def integrate(u, triangles_map):
    if inspect.isfunction(u):
        integrals = []
        for counter, triangle in triangles_map.items():
            centroid = 1/3 * np.sum(triangle, axis=0)
            if "triangle" in u.__code__.co_varnames:
                integrals.append(u(*centroid, triangle) * S(triangle)) 
            else:
                integrals.append(u(*centroid) * S(triangle))
        return sum(integrals)

    elif len(triangles_map.items()) <= 18: #If amount of triangles is low - comute as sum of integrals
        integrals = []
        triangles_num = len(triangles_map.items())
        x1, x2 = free_symbols(u)
        for counter, triangle in triangles_map.items():
            x1_m, x1_M, x2_m, x2_M = min(triangle[:, 0]), max(triangle[:, 0]), min(triangle[:, 1]), max(triangle[:, 1])
            x1_d, x2_d = x1_M - x1_m, x2_M - x2_m
            x1_ = [x1_m, x1_M]
            line = x2_d/x1_d*x1
            n_line = x2_d - line 
            x2_ = ([line                , x2_M                ]) if counter in (0, triangles_num - 2) else \
                  ([x2_m                , line                ]) if counter in (1, triangles_num - 1) else \
                  ([x2_m                , x1_m + x2_m + n_line]) if counter % 2 == 0                  else \
                  ([x1_m + x2_m + n_line, x2_M                ]) if counter % 2 == 1                  else None
            integrals.append(sp.integrate(u, [x2] + x2_, [x1] + x1_)) 
        return sum(integrals)
    
    else: #If amount of triangles is big compute integral on area at once
        triangles_stack = np.vstack(list(triangles_map.values()))
        X_RANGE, Y_RANGE = [min(triangles_stack[:,0]), max(triangles_stack[:,0])],\
                           [min(triangles_stack[:,1]), max(triangles_stack[:,1])]
        x1, x2 = free_symbols(u)
        return sp.integrate(u, [x1, *X_RANGE], [x2, *Y_RANGE])

def L_norm(u_sym, triangles_map):
    x = free_symbols(u_sym)
    return sp.sqrt(integrate(sp.lambdify(x, u_sym**2), triangles_map))

def W_norm(u_sym, triangles_map):
    x = free_symbols(u_sym)
    return sp.sqrt(integrate(sp.lambdify(x, u_sym**2 + sp.diff(u_sym, x[0])**2 + sp.diff(u_sym, x[1])**2), triangles_map))

def L_diff_norm(u_sym, u_h, triangles_map):
    x = free_symbols(u_sym)
    u_sym_minus_u_h = lambda X1, X2, triangle: (sp.lambdify(x, u_sym)(X1, X2) - u_h(X1, X2, triangle)[0])**2
    return sp.sqrt(integrate(u_sym_minus_u_h, triangles_map))

def W_diff_norm(u_sym, u_h, triangles_map):
    x = free_symbols(u_sym)
    u_sym_minus_u_h = lambda X1, X2, triangle: \
            (sp.lambdify(x, u_sym)(X1, X2)                - u_h(X1, X2, triangle)[0])**2 + \
            (sp.lambdify(x, sp.diff(u_sym, x[0]))(X1, X2) - u_h(X1, X2, triangle)[1])**2 + \
            (sp.lambdify(x, sp.diff(u_sym, x[1]))(X1, X2) - u_h(X1, X2, triangle)[2])**2 
    return sp.sqrt(integrate(u_sym_minus_u_h, triangles_map))

###Helpers
def free_symbols(f): #Extracts variables sorted by name from function (eg. f=2*x2+x1**2 ==> [x1, x2])
    return sorted(f.free_symbols, key=lambda s: s.name)

def subs(f, sub): #Passes elements along last axis into sympy function (eg. f=x1*x2 and sub=[[[1, 2],[3, 4]]] ==> [[2, 12]])
    if inspect.isfunction(f):
        func = lambda xs: f(*xs)
    else:
        func = lambda xs: f.subs(zip(free_symbols(f), xs))
    return np.apply_along_axis(func1d=func, axis=-1, arr=sub).astype(float)

#def grad(f): #Calculates sympy gradient (eg. f=x1**2+3*x2**2 ==> np.array([2, 6]))
#    return np.array([sp.diff(f, var, var) for var in free_symbols(f)])

###Output
def print_title(title, padding=10): #Prints formated text
    title_l = len(title)
    print_width = padding*2 + title_l 
    print('#'*print_width)
    print(f"{'#'*padding}{title}{'#'*padding}")
    print('#'*print_width)
