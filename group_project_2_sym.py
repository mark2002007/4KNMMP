import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

X_RANGE = (0, 2)
Y_RANGE = (0, 2)
#
#==========Part 1==========
#
def get_dots(dots_num, x_range = X_RANGE, y_range = Y_RANGE):
    dots_num_per_axis = int(np.sqrt(dots_num)) + 2
    dots = [[(x, y) for x in np.linspace(*x_range, dots_num_per_axis)] for y in np.linspace(*y_range, dots_num_per_axis)]
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

def draw_dots(dots_map, ax):
    for (name, coords) in dots_map.items():            
        ax.plot3D(*coords, 'ko')
        ax.annotate(name, [coords[0]-0.025, coords[1]+0.05])

def draw_triangles(triangles, ax):
    for name, triangle in triangles.items():
        xs, ys = zip(*triangle, triangle[0])
        ax.plot3D(xs, ys)
        ax.annotate(name, [np.average(xs), np.average(ys)])

def print_title(title, padding=10):
    title_l = len(title)
    print_width = padding*2 + title_l 
    print('#'*print_width)
    print(f"{'#'*padding}{title}{'#'*padding}")
    print('#'*print_width)

#
#==========Part 2==========
#
def free_symbols(f):
    return sorted(f.free_symbols, key=lambda s: s.name)

def grad(f):
    return np.array([[sp.diff(f, var)] for var in free_symbols(f)])

#
#=========For the future======
#
#def subs(arr, sub):
    #sub = list(sub)
    #return np.vectorize(lambda e: e.subs(sub))(arr)
#
#def N(arr):
    #return np.vectorize(lambda x: sp.N(x))(arr)
#
#
#
def main():
    #================================================== 
    #dots_num = 4 #N^2. Values n from [N^2, (N+1)^2] rounds to N^2
    #dots = get_dots(dots_num)
    #dots_map = get_dots_enumeration(dots)
    #dots_map_reverse = {tuple(v): k for k, v in dots_map.items()}
    #triangles = get_triangles(dots)
    #triangles_map = get_triangles_enumeration(triangles)

    #print_title("Dots")
    #print(dots)
    #print_title("Dots mapping")
    #for name, coords in dots_map.items():
        #print(f"\u001b[31m{name} : \u001b[0m{coords}")
    #print_title("Triangles To Coordinates")
    #for name, triangle in triangles_map.items():
        #print(f"\u001b[31m{name} : \u001b[0m\n{triangle}")
    #print_title("Trianlges To Dot Names")
    #for name, triangle in triangles_map.items():
        #print(f"\u001b[31m{name} : \u001b[0m[", end="")
        #for dot_coords in triangle[:-1]:
            #print(f"{dots_map_reverse[tuple(dot_coords)]},", end="")
        #print(f"{dots_map_reverse[tuple(triangle[-1])]}]")

    #fig = plt.figure()
    #ax = plt.axes(projection="3d")
    #ax.set_zlim(0, 2)
    #draw_triangles(triangles_map, ax)
    #draw_dots(dots_map, ax)    
    #==================================================

    x1, x2 = sp.symbols("x1:3")
    u = (x1-2)**2*(x2-2)**2*x1*x2
    c = np.array([-1, -1, 2])
    f = np.dot(c, np.vstack((grad(u), u))).item()
    print(f)
    sp.plotting.plot3d(f)

if __name__ == "__main__":
    main()
