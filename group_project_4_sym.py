from lib import *

X_RANGE = (0, 2)
Y_RANGE = (0, 2)

def main():
    #================================================== 
    dots_per_axis = 8
    dots = get_dots(dots_per_axis, X_RANGE, Y_RANGE)
    dots_num = dots.shape[0] * dots.shape[1]
    dots_map = get_dots_enumeration(dots)
    dots_map_reverse = {tuple(v): k for k, v in dots_map.items()}
    triangles = get_triangles(dots)
    triangles_map = get_triangles_enumeration(triangles)
    #================================================== 
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
    x = sp.symbols("x1:3") 
    x1, x2 = x
    u = (x1-2)**2*(x2-2)**2*x1*x2
    #u = x1**2 + x2**2
    #coefs = np.array([1, 1, 2]) # a_11, a_22, d
    a_11, a_22, d = 1, 1, 2
    f = (-a_11*grad(u)[0] -a_22*grad(u)[1] + u).item()
    print(f"[a_11, a_22, d] : {[a_11, a_22, d]}")
    print(f"f : {f}")
    A = np.zeros((dots_num,)*2)
    B = np.zeros((dots_num,1))
    for t in triangles:
        dot_nums = [dots_map_reverse[tuple(dot)] for dot in t] #Numbers of dots of current triangle
        #Find ABCe tensors
        delta = 2*S(t)
        Me = M(d, delta)
        Ke = K(t, a_11, a_22)
        Ae = Ke + Me
        #fe = np.array([subs(f, xi).item() for xi in t]).reshape((-1, 1))
        fe = subs(f, t).reshape((-1, 1))
        Qe = Q(Me, d, fe)  
        #Add Ae to A
        for i in range(Ae.shape[0]):
            for j in range(Ae.shape[1]):
                A[dot_nums[i], dot_nums[j]] += Ae[i, j]  
        #Add Qe to B
        for i in range(Qe.shape[0]):
            B[dot_nums[i], 0] += Qe[i, 0]
        #Some outputs
        print("#"*100)
        print(f"dot_nums :\n{dot_nums}", end="\n\n")
        print(f"Me :\n{Me}", end="\n\n")
        print(f"Ke :\n{Ke}", end="\n\n")
        print(f"fe :\n{fe}", end="\n\n")
        print(f"Ae :\n{Ae}", end="\n\n")
        print(f"A  :\n{A}" , end="\n\n")
        print(f"Qe :\n{Qe}", end="\n\n")
        print(f"B  :\n{B}" , end="\n\n")
    #Solve Au_h=b
    u_h = np.linalg.solve(A, B).reshape((dots_per_axis,)*2)
    print("u_h")
    print(u_h)
    print("u")
    #print(dots)
    u_acc = subs(u, dots)
    print(u_acc)
    #
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='3d'))
    ax.plot_surface(dots[...,0], dots[...,1], u_acc)
    ax.plot_surface(dots[...,0], dots[...,1], u_h)
    plt.show()

if __name__ == "__main__":
    main()


