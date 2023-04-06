from lib import *

X_RANGE = (0, 2)
Y_RANGE = (0, 2)
VERBOSE = False

def main():
    #================================================== 
    dots_per_axis = 4
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
    f_sym = -a_11*sp.diff(u_sym, x1, x1) -a_22*sp.diff(u_sym, x2, x2) + d*u_sym
    #================================================== 
    beta  = np.array([1e-10, 1e-10, 1e-10, 1e-10])
    sigma = np.array([1e+10, 1e+10, 1e+10, 1e+10])
    psi   = np.array([0, 0, 0, 0])
    Gamma = np.array([X_RANGE[1] - X_RANGE[0], Y_RANGE[1] - Y_RANGE[0]])/(dots_per_axis-1)
    #
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
    u_h_ = []
    q = np.linalg.solve(A, B).reshape([-1, 1]) 
    
    def u_h(X):
        X1, X2 = X
        print(f"in_t_0 : in_tr")
        print(f"X : {X}")
        input(f"triangle id : {in_which_triangle([X1, X2], triangles)}")
        triangle = triangles[in_which_triangle([X1, X2], triangles)]
        dot_nums = [dots_map_reverse[tuple(dot)] for dot in triangle] 
        N_e = (get_a_b_c(triangle).T @ np.array([1, X1, X2])[:, None]).flatten()
        q_e = q[dot_nums, :]
        return (N_e@q_e).item()
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(dots[...,0], dots[...,1], subs(u, dots))
    ax.plot_surface(dots[...,0], dots[...,1], q.reshape((dots_per_axis,)*2))
    ax.plot_surface(dots[...,0], dots[...,1], np.apply_along_axis(u_h, -1, dots))
    plt.show()
    input()
#    u_h = lambda X1, X2:  
#    for _, triangle in enumerate(triangles):
#        dot_nums = [dots_map_reverse[tuple(dot)] for dot in triangle] #Numbers of dots of current triangle
#        q_e = q[dot_nums, :]
#        N_e = (get_a_b_c(triangle).T@np.array([1, x1, x2])[:, None]).flatten()
#        u_h_.append((N_e@q_e).item())
#    u_h_ = np.array(u_h_)
#    print(*u_h_, sep="\n")
#    u_h = sp.lambdify([x1, x2], u_h_[in_which_triangle([x1, x2], triangles)])

    input()
    u_acc = subs(u, dots).reshape([-1, 1])
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


