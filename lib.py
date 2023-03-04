import sys
import numpy as np
np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(threshold=sys.maxsize)
import sympy as sp
import matplotlib.pyplot as plt

###Triangulation
def get_dots(dots_per_axis, x_range, y_range): #Calculates matrix of dots where each dots is vector of x and y position
    dots = [[(x, y) for x in np.linspace(*x_range, dots_per_axis)] for y in np.linspace(*y_range, dots_per_axis)]
    return np.array(dots)

def get_dots_enumeration(dots):
    return dict(enumerate(dots.reshape((-1, dots.shape[-1]))))

def get_triangles(dots):
    triangles = []
    for yi in range(len(dots) - 1):
        for xi in range(len(dots[yi]) - 1):
            if (xi, yi) in [(0, 0), (len(dots[xi])-2, len(dots[xi])-2)]:
                triangles.append([dots[yi, xi], dots[yi+1, xi+1], dots[yi+1, xi]])
                triangles.append([dots[yi, xi], dots[yi, xi+1], dots[yi+1, xi+1]])
            else:
                triangles.append([dots[yi+1, xi], dots[yi, xi], dots[yi, xi+1]])
                triangles.append([dots[yi, xi+1], dots[yi+1, xi+1], dots[yi+1, xi]])
    return np.array(triangles)

def get_triangles_enumeration(triangles):
    return dict(enumerate(triangles))

###Draw
def draw_dots(dots_map, ax):
    for name, coords in dots_map.items():            
        ax.plot(*coords, 'ko')
        ax.annotate(name, [coords[0]-0.025, coords[1]+0.05])

def draw_triangles(triangles_map, ax):
    for name, triangle in triangles_map.items():
        xs, ys = zip(*triangle, triangle[0])
        ax.plot(xs, ys)
        ax.annotate(name, [np.average(xs), np.average(ys)])

def print_title(title, padding=10): #Prints formated text
    title_l = len(title)
    print_width = padding*2 + title_l 
    print('#'*print_width)
    print(f"{'#'*padding}{title}{'#'*padding}")
    print('#'*print_width)

###Helpers
def free_symbols(f): #Extracts variables from function sorted by name (eg. f=2*x2+x1**2 ==> [x1, x2])
    return sorted(f.free_symbols, key=lambda s: s.name)

def subs(f, sub): #Passes elements along last axis into sympy function (eg. f=x1*x2 and sub=[[[1, 2],[3, 4]]] ==> [[2, 12]])
    func = lambda xs: f.subs(zip(free_symbols(f), xs))
    return np.apply_along_axis(func1d=func, axis=-1, arr=sub) 

def grad(f): #Calculates sympy gradient (eg. f=x1**2+3*x2**2 ==> np.array([2, 6]))
    return np.array([[sp.diff(f, var, var)] for var in free_symbols(f)])

def S(triangle): #Calculates area of triangle (eg. triangle=np.array([[0, 0],[1, 0],[0, 1]]) ==> 0.5)
    matrix = np.hstack((np.ones((3, 1)), triangle))
    area = 0.5*np.linalg.det(matrix)
    return area

def R(sigma, beta, Gamma): 
    return (sigma*Gamma)/(beta*6)*(np.eye(2)+1)

def P(psi, beta, Gamma):
    return (psi*Gamma)/(beta*2)*np.array([[1],[1]])

def M(d, delta): #Calculates M_e matrix (PS. np.eye(3) + 1 returns np.array([[2,1,1],[1,2,1],[1,1,2]]))
    return (delta/24)*(np.eye(3)+1)

def get_b_c(triangle): #Calculates b_i, b_j, b_m, c_i, c_j, c_m coeficients for triangle
    t = triangle
    i, j, m = range(3)
    return np.array([
        [t[j, 1] - t[m, 1], t[m, 0] - t[j, 0]],
        [t[m, 1] - t[i, 1], t[i, 0] - t[m, 0]],
        [t[i, 1] - t[j, 1], t[j, 0] - t[i, 0]],
        ])

def K(triangle, a_11, a_22): #Calculates K_e matrix
    b, c = get_b_c(triangle).T
    b_i, b_j, b_m = b
    c_i, c_j, c_m = c
    return (a_11*np.array([
            [b_i**2 , b_i*b_j, b_i*b_m],
            [b_i*b_j, b_j**2 , b_j*b_m],
            [b_i*b_m, b_j*b_m, b_m**2 ],
            ])
            +
            a_22*np.array([
            [c_i**2 , c_i*c_j, c_i*c_m],
            [c_i*c_j, c_j**2 , c_j*c_m],
            [c_i*c_m, c_j*c_m, c_m**2 ],
            ]))/(4*S(triangle))

def Q(Me, d, fe): #Calculates Q_e matrix
    return np.dot(Me/d, fe)

