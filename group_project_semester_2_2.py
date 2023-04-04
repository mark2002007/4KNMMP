from lib import *

X_RANGE = (0, 2)
Y_RANGE = (0, 2)
VERBOSE = False

def main():
    #================================================== 
    dots_per_axis = 3
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
    u_sym = (x1-2)**2*(x2-2)**2*x1*x2
#    u_sym = x1**2 + x2**2
#    f_sym = -a_11*grad(u_sym)[0] -a_22*grad(u_sym)[1] + d*u_sym
    #================================================== 
    print(f"INTEGRAL (sum) : {integrate(u_sym, triangles_map)}")
    print(f"INTEGRAL : {sp.integrate(u_sym, [x2, *Y_RANGE], [x1, *X_RANGE])}")
    input()
    quit()
    #================================================== 
    beta  = np.array([1e-10, 1e-10, 1e-10, 1e-10])
    sigma = np.array([1e+10, 1e+10, 1e+10, 1e+10])
    psi   = np.array([0, 0, 0, 0])
    Gamma = np.array([X_RANGE[1] - X_RANGE[0], Y_RANGE[1] - Y_RANGE[0]])/(dots_per_axis-1)
    #
    print(f"[a_11, a_22, d] : {[a_11, a_22, d]}")
    print(f"f : {f}")
    u = sp.lambdify(x, u_sym)
    f = sp.lambdify(x, f_sym)
    A = np.zeros((dots_num, dots_num))
    B = np.zeros((dots_num,1))
    delta = 2*S(triangles[0])
    for counter, triangle in enumerate(triangles):
        print(f"{counter+1}/{triangles_num}")
        dot_nums = [dots_map_reverse[tuple(dot)] for dot in triangle] #Numbers of dots of current triangle
        #FEM matrices
        Ke = K(triangle, delta, a_11, a_22)
        Me = M(d, delta)
        Ae = Ke + Me
        fe = subs(f, triangle)[:, None]
        Qe = Q(Me, d, fe)
        #Add Ae to A and Qe to B
        A[np.ix_(dot_nums, dot_nums)] += Ae
        B[dot_nums, 0] += Qe[:, 0]
        #
        boundary_dots = extract_boundary_dots(triangle, X_RANGE, Y_RANGE)
        if is_boundary_el(boundary_dots): 
            boundary_dot_nums = [dots_map_reverse[tuple(dot)] for dot in boundary_dots]
            boundary_num = which_boundary(boundary_dots, X_RANGE, Y_RANGE)
            #Bonus FEM matrices
            Re = R(sigma[boundary_num], beta[boundary_num], Gamma[boundary_num%2])
            Pe = P(psi[boundary_num]  , beta[boundary_num], Gamma[boundary_num%2])
            #Add Re to A and Pe to B
            A[np.ix_(boundary_dot_nums, boundary_dot_nums)] += Re
            B[boundary_dot_nums, 0] += Pe[:, 0]
        #Some outputs
        if VERBOSE:
            print("#"*100)
            print(f"dot_nums :\n{dot_nums}", end="\n\n")
            print(f"Me :\n{Me}", end="\n\n")
            print(f"Ke :\n{Ke}", end="\n\n")
            print(f"Ae :\n{Ae}", end="\n\n")
            print(f"A  :\n{A}" , end="\n\n")
            print(f"fe :\n{fe}", end="\n\n")
            print(f"Qe :\n{Qe}", end="\n\n")
            print(f"B  :\n{B}" , end="\n\n")
    #Solve Au_h=b
    u_h = np.linalg.solve(A, B).reshape((dots_per_axis,)*2)
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


