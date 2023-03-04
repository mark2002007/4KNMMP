import numpy as np
import matplotlib.pyplot as plt

X_RANGE = (0, 2)
Y_RANGE = (0, 2)

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


def notched(x, dx, var_i):
    x_copy = np.copy(x)
    x_copy[..., var_i] += dx
    return x_copy

def diff(f, x, var_i=0, order=1, Dx=1e-8):
    if order == 0:
        return f(x)
    else:
        #print(f"{var_i}, {order}, {Dx} : x+Dxs = {x.shape} + {Dxs.shape} = {(x+Dxs).shape}")
        return (diff(f, notched(x, Dx, var_i), var_i, order-1, Dx/2) - diff(f, x, var_i, order-1, Dx/2))/Dx

def grad(f, x, order=1):
    return np.array([diff(f, x, var_i, order) for var_i in range(x.shape[-1])])

def main():

    dots_num = 4 #N^2. Values n from [N^2, (N+1)^2] rounds to N^2
    dots = get_dots(dots_num)
    dots_map = get_dots_enumeration(dots)
    dots_map_reverse = {tuple(v): k for k, v in dots_map.items()}
    triangles = get_triangles(dots)
    triangles_map = get_triangles_enumeration(triangles)

    #=========================PRINING=========================
    print_title("Dots")
    print(dots)
    print_title("Dots mapping")
    for name, coords in dots_map.items():
        print(f"\u001b[31m{name} : \u001b[0m{coords}")
    print_title("Triangles To Coordinates")
    for name, triangle in triangles_map.items():
        print(f"\u001b[31m{name} : \u001b[0m\n{triangle}")
    print_title("Trianlges To Dot Names")
    for name, triangle in triangles_map.items():
        print(f"\u001b[31m{name} : \u001b[0m[", end="")
        for dot_coords in triangle[:-1]:
            print(f"{dots_map_reverse[tuple(dot_coords)]},", end="")
        print(f"{dots_map_reverse[tuple(triangle[-1])]}]")
    #=========================================================
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_zlim(0, 2)
    draw_triangles(triangles_map, ax)
    draw_dots(dots_map, ax)
    
    u = lambda x: (x[...,0] - 2)**2*(x[...,1])**2*x[...,0]*x[...,1]
    grid = np.mgrid[0:2:0.01, 0:2:0.01]
    xs = grid.reshape([-1, 1]) if len(grid.shape) == 1 else np.transpose(grid, axes=[*np.arange(len(grid.shape))[1:],0])
    c = np.array([-1, -1, 2]).reshape([-1,1,1])
    f = lambda x: np.sum(c*np.array([diff(u, xs, 0, 2), diff(u, xs, 1, 2), u(x)]), axis=0)
    #f = lambda x: x[...,0]**2 + x[...,1]**2
    print(f"grid.shape : {grid.shape}; xs.shape : {xs.shape}, u(xs).shape : {u(xs).shape}")
    print(f"f(xs).shape : {f(xs).shape}")
    ax.plot_surface(xs[...,0], xs[...,1], f(xs))
    plt.show()

if __name__ == "__main__":
    main()
