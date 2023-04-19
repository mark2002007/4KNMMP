from lib import *

X_RANGE = (0, 2)
Y_RANGE = (0, 2)
VERBOSE = False

def main():
    #================================================== 
    dots_per_axis = 16 + 8
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
    q = np.linalg.solve(A, B).reshape([-1, 1]) 
    
    def u_h(X1, X2, triangle = None):
        if triangle is None:
            triangle = triangles[in_which_triangle([X1, X2], triangles_map)]

        a_b_c = get_a_b_c(triangle)
        N_e = 1/delta * (a_b_c @ np.array([1, X1, X2])[:, None]).flatten()
        dot_nums = [dots_map_reverse[tuple(dot)] for dot in triangle]
        q_e = q[dot_nums, :]
        
        u_h_val = (N_e@q_e).item()
        u_h_x1 = (a_b_c.T[1]/delta @ q_e).item()
        u_h_x2 = (a_b_c.T[2]/delta @ q_e).item()
        return u_h_val, u_h_x1, u_h_x2 
    
#    input(f"I(u_h) : {integrate(lambda *x: u_h(*x)[0], triangles_map)}")
#    input(f"I(u)   : {integrate(u, triangles_map)}")
#    input(f"I(u - u_h) : {integrate(lambda X1, X2, triangle: u(X1, X2) - u_h(X1, X2, triangle)[0], triangles_map)}")
    input(f"L_norm(u - u_h) : {L_diff_norm(u_sym, u_h, triangles_map)}")
    input(f"W_norm(u - u_h) : {W_diff_norm(u_sym, u_h, triangles_map)}")
    input(f"L_norm(u - u_h) / L_norm(u) : {L_diff_norm(u_sym, u_h, triangles_map)/L_norm(u_sym, triangles_map)}")
    input(f"W_norm(u - u_h) / W_norm(u) : {W_diff_norm(u_sym, u_h, triangles_map)/W_norm(u_sym, triangles_map)}")

if __name__ == "__main__":
    main()


