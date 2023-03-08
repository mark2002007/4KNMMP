import sys
import itertools as it
import numpy as np
np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(threshold=sys.maxsize)
import sympy as sp
import matplotlib.pyplot as plt

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
                triangles.append([dots[yi, xi], dots[yi+1, xi+1], dots[yi+1, xi  ]])
                triangles.append([dots[yi, xi], dots[yi, xi+1]  , dots[yi+1, xi+1]])
            else:
                triangles.append([dots[yi+1, xi], dots[yi, xi]    , dots[yi, xi+1]])
                triangles.append([dots[yi, xi+1], dots[yi+1, xi+1], dots[yi+1, xi]])
    return np.array(triangles)

def get_triangles_map(triangles): #Generages mapping from triangle number to triangle
    return dict(enumerate(triangles))

def inner_outer_dot_nums(dots, dots_map, dots_map_reverse):
    inner_dot_nums = [dots_map_reverse[tuple(dot)] for dot in dots[1:-1, 1:-1].reshape(-1, 2)]
    outer_dot_nums = list(set(range(dots.shape[0] * dots.shape[1])) - set(inner_dot_nums))
    return inner_dot_nums, outer_dot_nums

###Output
def print_title(title, padding=10): #Prints formated text
    title_l = len(title)
    print_width = padding*2 + title_l 
    print('#'*print_width)
    print(f"{'#'*padding}{title}{'#'*padding}")
    print('#'*print_width)

###Matrices
def free_symbols(f): #Extracts variables sorted by name from function (eg. f=2*x2+x1**2 ==> [x1, x2])
    return sorted(f.free_symbols, key=lambda s: s.name)

def subs(f, sub): #Passes elements along last axis into sympy function (eg. f=x1*x2 and sub=[[[1, 2],[3, 4]]] ==> [[2, 12]])
    func = lambda xs: f.subs(zip(free_symbols(f), xs))
    return np.apply_along_axis(func1d=func, axis=-1, arr=sub).astype(np.float64)

def grad(f): #Calculates sympy gradient (eg. f=x1**2+3*x2**2 ==> np.array([2, 6]))
    return np.array([sp.diff(f, var, var) for var in free_symbols(f)])

def S(triangle): #Calculates area of triangle (eg. triangle=np.array([[0, 0],[1, 0],[0, 1]]) ==> 0.5)
    matrix = np.hstack((np.ones((3, 1)), triangle))
    return .5*np.linalg.det(matrix)

def get_b_c(triangle): #Calculates b_i, b_j, b_m, c_i, c_j, c_m coeficients for triangle
    t = triangle
    i, j, m = range(3)
    x, y = range(2)
    return np.array([
        [t[j, y] - t[m, y], t[m, y] - t[i, y], t[i, y] - t[j, y]],
        [t[m, x] - t[j, x], t[i, x] - t[m, x], t[j, x] - t[i, x]],
        ])

def K(triangle, delta, a_11, a_22): #Calculates K_e matrix
    b, c = get_b_c(triangle)
    return (1/(2*delta)) * (a_11*b[:,None]@b[None,:]+a_22*c[:,None]@c[None,:])

def M(d, delta): #Calculates M_e matrix (btw. np.eye(3) + 1 returns np.array([[2,1,1],[1,2,1],[1,1,2]]))
    return (d*delta/24)*(np.eye(3)+1)

def Q(Me, d, fe): #Calculates Q_e matrix
    return Me/d@fe

