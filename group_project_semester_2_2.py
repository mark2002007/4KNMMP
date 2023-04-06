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
#    u_sym = (x1-2)**2*(x2-2)**2*x1*x2
    u_sym = x1**2 + x2**2
#    f_sym = -a_11*sp.diff(u_sym, x1, x1) -a_22*sp.diff(u_sym, x2, x2) + d*u_sym
    #================================================== 
    print(f"INTEGRAL (sum) : {integrate(u_sym, triangles_map)}")
    print(f"INTEGRAL : {sp.integrate(u_sym, [x2, *Y_RANGE], [x1, *X_RANGE])}")
    print(f"L_norm : {L_norm(u_sym, triangles_map)}")
    print(f"W_norm : {W_norm(u_sym, triangles_map)}")
    input()
    quit()
    #================================================== 

if __name__ == "__main__":
    main()


