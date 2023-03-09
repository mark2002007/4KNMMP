from lib import *

X_RANGE = (0, 2)
Y_RANGE = (0, 2)
VERBOSE = False

def main():
    #================================================== 
    dots_per_axis = 17
    dots = get_dots(dots_per_axis, X_RANGE, Y_RANGE)
    dots_num = dots.shape[0] * dots.shape[1]
    dots_map              = get_dots_map(dots)
    dots_map_reverse      = {tuple(dot): dot_num for dot_num, dot in dots_map.items()}#
    triangles             = get_triangles(dots)
    triangles_num         = len(triangles)
    triangles_map         = get_triangles_map(triangles)
    triangles_map_reverse = {tuple(map(tuple, triangle)) : triangle_num for triangle_num, triangle in triangles_map.items()}
    #================================================== 
    x1, x2 = x = sp.symbols("x1:3") 
    a_11, a_22, d = 1, 1, 2
    u = (x1-2)**2*(x2-2)**2*x1*x2
    #u = (x1 - 1)*(x2 - 1)*x1*x2**2
    f = -a_11*grad(u)[0] -a_22*grad(u)[1] + d*u
    print(f"[a_11, a_22, d] : {[a_11, a_22, d]}")
    print(f"f : {f}")
    u = sp.lambdify(x, u)
    f = sp.lambdify(x, f)
    A = np.zeros((dots_num, dots_num))
    B = np.zeros((dots_num,1))
    delta = 2*S(triangles[0])
    for counter, triangle in enumerate(triangles):
        print(f"{counter+1}/{triangles_num}")
        dot_nums = [dots_map_reverse[tuple(dot)] for dot in triangle] #Numbers of dots of current triangle
        #Find ABCe tensors
        Ke = K(triangle, delta, a_11, a_22)
        Me = M(d, delta)
        Ae = Ke + Me
        fe = subs(f, triangle)[:, None]
        Qe = Q(Me, d, fe)  
        #Add Ae to A and Qe to B
        A[np.ix_(dot_nums, dot_nums)] += Ae
        B[dot_nums, 0] += Qe[:, 0]
        #Some outputs
        if VERBOSE:
            print("#"*100)
            print(f"dot_nums :\n{dot_nums}", end="\n\n")
            print(f"Me :\n{Me}", end="\n\n")
            print(f"Ke :\n{Ke}", end="\n\n")
            print(f"fe :\n{fe}", end="\n\n")
            print(f"Ae :\n{Ae}", end="\n\n")
            print(f"A  :\n{A}" , end="\n\n")
            print(f"Qe :\n{Qe}", end="\n\n")
            print(f"B  :\n{B}" , end="\n\n")
    
    _, outer_dot_nums = inner_outer_dot_nums(dots, dots_map, dots_map_reverse)
    A = np.delete(A, outer_dot_nums, 0)
    A = np.delete(A, outer_dot_nums, 1)
    B = np.delete(B, outer_dot_nums, 0)
    #Solve Au_h=b
    u_h = np.linalg.solve(A, B).reshape((dots_per_axis-2,)*2)
    u_h = np.pad(u_h, 1)
    u_acc = subs(u, dots)
    print("u_h")
    print(u_h)
    print("u")
    print(u_acc)
    print(f"norm : {np.linalg.norm(u_h - u_acc)}")
    #
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='3d'))
    ax.plot_surface(dots[...,0], dots[...,1], u_acc, alpha=.6)
    ax.plot_surface(dots[...,0], dots[...,1], u_h, alpha=.6)
    plt.show()

if __name__ == "__main__":
    main()


