from structure.point import Point
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp
import swiglpk as glp

def leastsq(R, c, **kwargs):
    '''
    https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
    min_x norm(Rx-c) s.t. Gx <= h

    https://scaron.info/doc/qpsolvers/supported-solvers.html#module-qpsolvers.solvers.osqp_
    by default,
    eps_abs, eps_rel = 1e-5
    max_iter = 4000

    :return: optimal objective value
    '''
    # Create an OSQP object

    return solve_qp(R, c, **kwargs, solver='osqp')


class PointSet:
    def __init__(self, input_filename=None, P=None):
        if input_filename is None and P is None:
            self.points = []
        elif input_filename is None and P is not None:
            self.points = []
            for pp in P:
                self.points.append(pp)
        elif input_filename is not None and P is None:
            filename = f"./input/{input_filename}"
            # print(filename)
            try:
                with open(filename, 'r') as file:
                    number_of_points, dim = map(int, file.readline().split())
                    self.points = []
                    for _ in range(number_of_points):
                        coords = np.array(list(map(float, file.readline().split())))
                        if len(coords) != dim:
                            raise ValueError("The dimension of the points does not match the specified dimension.")
                        point = Point(dim, id=len(self.points), coord=coords)
                        self.points.append(point)
            except FileNotFoundError:
                print(f"Cannot open the data file {filename}.")
                exit(0)
            except ValueError as e:
                print(f"Error reading the data file: {e}")
                exit(0)

    def add_point(self, point):
        self.points.append(point)

    # find the skyline points
    def skyline(self):
        skyline_indices = []
        for i, pt in enumerate(self.points):
            if i % 100 == 0:
                print(i)
            dominated = False
            # Check if pt is dominated by the skyline so far
            for idx in skyline_indices:
                if self.points[idx].dominates(pt):
                    dominated = True
                    break

            if not dominated:
                # Eliminate any points in the current skyline that it dominates
                new_skyline_indices = []
                for idx in skyline_indices:
                    if not pt.dominates(self.points[idx]):
                        new_skyline_indices.append(idx)
                new_skyline_indices.append(i)
                skyline_indices = new_skyline_indices

        # Create the skyline set
        skyline = PointSet()
        for idx in skyline_indices:
            skyline.add_point(self.points[idx])

        return skyline

    # find the top-k points
    def find_top_k(self, u, k):
        top = []
        value = []

        # Set the initial k points
        top.append(self.points[0])
        value.append(u.dot_prod(self.points[0]))
        for j in range(1, k):
            sum0 = u.dot_prod(self.points[j])
            position = 0
            for z in range(len(value)):
                if sum0 > value[z]:
                    break
                position = z + 1
            top.insert(position, self.points[j])
            value.insert(position, sum0)

        # Adjust the list for all remaining points in the set
        for j in range(k, len(self.points)):
            sum0 = u.dot_prod(self.points[j])
            position = k + 1
            for z in range(k, 0, -1):
                if sum0 <= value[z - 1]:
                    break
                position = z - 1
            if position < k:
                top.insert(position, self.points[j])
                value.insert(position, sum0)
                top.pop()
                value.pop()

        return top

    def print(self):
        for point in self.points:
            point.print()

    def subtract(self, pset):
        for p in pset.points:
            for q in self.points:
                if q.is_same(p):
                    self.points.remove(q)
                    break

    # QP solver
    def prune_cone(self, x: Point, epsilon) -> bool:
        S, s = self.points, len(self.points)
        if s <= 1:
            return False

        #print(x.coord, x.dot_prod(Point(x.dim, coord=[0.25] * x.dim)))
        #print("S:    ")
        #for pp in S:
        #    print(pp.coord, pp.dot_prod(Point(x.dim, coord=[0.25] * x.dim)))

        # Run constrained least-squares below
        # min_x norm(Rx-c) s.t. Gx <= h
        #R = [S[i - 1].coord - S[i].coord for i, _ in enumerate(S) if i > 0]
        #R = np.array(R).T
        #c = np.array(x.coord - S[0].coord)
        #lb = np.zeros(s - 1)

        # print(len(S))
        A = [S[i - 1].coord - S[i].coord for i, _ in enumerate(S) if i > 0]
        A = np.array(A)
        A = A.T
        b = np.array(x.coord - S[0].coord)
        b.reshape((-1, 1))
        P = np.dot(A.T, A)
        q = np.dot(b.T, A)
        h = np.zeros(s-1)

        est = solve_qp(P, q, lb=h, solver='osqp') #  leastsq(R, c, lb=lb, max_iter=4000)
        # print(est)
        if est is None:  # fails to solve
            return False

        term1 = np.dot(np.array(est), np.dot(A.T, np.dot(A, np.array(est).T)))
        term2 = 2 * np.dot(b.T, np.dot(A, np.array(est).T))
        term = term1 + term2
        a1 = np.dot(A, est) + b
        a = np.linalg.norm(A @ est + b)
        return a < epsilon + 0.001

    def is_prune(self, p: Point):
        M = len(self.points)
        if M < 1:
            return False
        if M == 1:
            return self.points[0].dominates(p)
        D = self.points[0].dim

        lp = glp.glp_create_prob()
        glp.glp_set_prob_name(lp, "is_prune")
        glp.glp_set_obj_dir(lp, glp.GLP_MAX)

        # add D rows: q_1, ..., q_D
        glp.glp_add_rows(lp, D)
        for i in range(1, D+1):
            glp.glp_set_row_name(lp, i, f"q{i}")
            glp.glp_set_row_bnds(lp, i, glp.GLP_LO, p.coord[i - 1] - self.points[0].coord[i - 1], 0)

        # add D columns: v_1, ...v_{M - 1}
        glp.glp_add_cols(lp, M - 1)
        for i in range(1, M):
            glp.glp_set_col_name(lp, i, f"v{i}")
            glp.glp_set_col_bnds(lp, i, glp.GLP_LO, 0.0, 0.0)  # 0 <= v[i] < infty

        ia = glp.intArray(1 + D * (M - 1))
        ja = glp.intArray(1 + D * (M - 1))
        ar = glp.doubleArray(1 + D * (M - 1))
        counter = 1
        for i in range(1, D):
            for j in range(1, M - 1):
                ia[counter] = i
                ja[counter] = j
                ar[counter] = self.points[j].coord[i-1] - self.points[j-1].coord[i-1]
                counter += 1

        # loading data
        glp. glp_load_matrix(lp, counter - 1, ia, ja, ar)
        # running simplex
        parm = glp.glp_smcp()
        glp.glp_init_smcp(parm)
        parm.msg_lev = glp.GLP_MSG_OFF
        # Solving the LP
        glp.glp_simplex(lp, parm)

        status = glp.glp_get_prim_stat(lp)

        glp.glp_delete_prob(lp)

        if status == glp.GLP_UNDEF or status == glp.GLP_FEAS:
            # printf("LP feasible error.\n");
            return True
        else:
            return False

'''
x = Point(2, coord=np.array([2, 2]))
ps = PointSet()
x3 = Point(2, coord=np.array([3, 3]))
x2 = Point(2, coord=np.array([4, 4]))
x1 = Point(2, coord=np.array([1, 1]))
ps.points.append(x3)
ps.points.append(x2)
ps.points.append(x1)
ps.prune_cone(x, 0.0)
'''