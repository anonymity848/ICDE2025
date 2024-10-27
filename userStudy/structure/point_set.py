import os
import json
from structure.point import Point
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp
import swiglpk as glp
import websockets

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

    def printMiddleSelection(self, num_question, alg_name, p1, p2, selection, session_id):
        folder_path = f"../result"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_path += f"/{session_id}.txt"

        if num_question == 1:
            with open(f"{folder_path}", "a") as out_cp:  # "a" represents adding to the end of the file
                out_cp.write(f"{alg_name}\n\n")

        d = p1.dim
        with open(f"{folder_path}", "a") as out_cp:  # "a" represents adding to the end of the file
            out_cp.write(f"Question {num_question}: \n")
            col_width = 15
            horizontal_border = "+" + "+".join(["-" * col_width for _ in range(d + 1)]) + "+\n"
            header = "|" + "Car".center(col_width) + "|" + "Year".center(col_width) + "|" + "Price".center(
                    col_width) + "|" + "Mileage".center(col_width) + "|" + "Tax".center(col_width) + "|" + "MPG".center(
                col_width) + "|" + "EngineSize".center(col_width) + "|\n"

            # 输出边框和表头
            out_cp.write(horizontal_border)
            out_cp.write(header)
            out_cp.write(horizontal_border)

            pset_org = PointSet(f'audi_orgSkyline.txt')
            p1 = pset_org.points[p1.id]
            p2 = pset_org.points[p2.id]

            # 输出每个 Tuple 的内容，使用相同的宽度并居中对齐
            tuple_1 = "|" + "1".center(col_width) + "|" + "|".join(
                [str(p1.coord[i]).center(col_width) for i in range(d)]) + "|\n"
            tuple_2 = "|" + "2".center(col_width) + "|" + "|".join(
                [str(p2.coord[i]).center(col_width) for i in range(d)]) + "|\n"

            out_cp.write(tuple_1)
            out_cp.write(horizontal_border)
            out_cp.write(tuple_2)
            out_cp.write(horizontal_border)
            out_cp.write(f"The user selects Point {selection} as his/her preferred one.\n\n")

    async def printFinal(self, result_point, num_question, websocket, session_id):
        folder_path = f"../result"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_path += f"/{session_id}.txt"

        d = result_point.dim
        with (open(f"{folder_path}", "a") as out_cp):  # "a" represents adding to the end of the file
            out_cp.write(f"\nThe finally returned tuple: \n")
            col_width = 15
            horizontal_border = "+" + "+".join(["-" * col_width for _ in range(d + 1)]) + "+\n"
            header = "|" + "Car".center(col_width) + "|" + "Year".center(col_width) + "|" + "Price".center(
                col_width) + "|" + "Mileage".center(col_width) + "|" + "Tax".center(col_width) + "|" + "MPG".center(
                col_width) + "|" + "EngineSize".center(col_width) + "|\n"

            # 输出边框和表头
            out_cp.write(horizontal_border)
            out_cp.write(header)
            out_cp.write(horizontal_border)

            pset_org = PointSet(f'audi_orgSkyline.txt')
            result_point = pset_org.points[result_point.id]

            # 输出每个 Tuple 的内容，使用相同的宽度并居中对齐
            tuple_1 = "|" + "Tuple".center(col_width) + "|" + "|".join(
                [str(result_point.coord[i]).center(col_width) for i in range(d)]) + "|\n"
            out_cp.write(tuple_1)
            out_cp.write(horizontal_border)
            out_cp.write(f"The total number of questions asked: {num_question} \n")

        # satisfiaction
        message = {
            "message": "Please rate your satisfaction with the returned cars on a scale from 1 to 10. \n"
                        "Here, 1 means you are least satisfied and 10 means you are most satisfied.\n\n"
                        "请按照 1 到 10 的评分标准来评价您对返回车辆的满意度。\n其中 1 表示最不满意，10 表示最满意。",
            "data_group_1": result_point.coord.tolist()
        }
        # 向客户端发送数据
        await websocket.send(json.dumps(message))

        # 等待客户端输入
        try:
            # 接收来自客户端的消息
            response = await websocket.recv()

            # 解析 JSON 字符串
            data = json.loads(response)
            if "score" in data:
                score = data["score"]
                print(f"Received score from client: {score}")
                with (open(f"{folder_path}", "a") as out_cp):
                    out_cp.write(f"The satisfaction score is: {score} \n")
            else:
                print("No valid score found in response")
        except json.JSONDecodeError:
            # 处理 JSON 解析错误
            print("Failed to decode JSON")
        except ValueError:
            # 处理无效输入
            print("Wrong Value")
        except websockets.ConnectionClosed:
            print("Client disconnected")


        # boredom
        boredness_present = ""
        boredness_present += f"The number of questions you asked: {num_question}\n"
        boredness_present += (f"Please give a number from 1 to 10 to indicate how bored you feel \n"
                              f"based on the number of questions you answered and this returned car.\n"
                              f"Here 10 denotes you feel the most bored and 1 denotes you feel the least bored.\n\n")

        boredness_present += (f"这一轮您一共被问了{num_question}个问题。\n"
                              f"请您根据推荐的车和问题个数为本轮的无聊程度打分，范围是 1 到 10。\n")
        boredness_present += "其中10 表示您感到最无聊，1 表示您感到最不无聊。\n"
        message = {
            "message": boredness_present,
            "data_group_1": result_point.coord.tolist(),
            "integer_value": num_question
        }
        # 向客户端发送数据
        await websocket.send(json.dumps(message))

        # 等待客户端输入
        try:
            # 接收来自客户端的消息
            response = await websocket.recv()

            # 解析 JSON 字符串
            data = json.loads(response)
            if "score" in data:
                score = data["score"]
                print(f"Received score from client: {score}")
                with (open(f"{folder_path}", "a") as out_cp):
                    out_cp.write(f"The boredom score is: {score} \n\n\n")
            else:
                print("No valid score found in response")
        except json.JSONDecodeError:
            # 处理 JSON 解析错误
            print("Failed to decode JSON")
        except ValueError:
            # 处理无效输入
            print("Wrong Value")
        except websockets.ConnectionClosed:
            print("Client disconnected")

    async def question_for_interaction(self, websocket, p1, p2, num_question):
        pset_org = PointSet(f'audi_orgSkyline.txt')
        p1 = pset_org.points[p1.id]
        p2 = pset_org.points[p2.id]

        # 创建要发送的数据字典
        message = {
            "message": "Please select the car you prefer. \n请选择您更喜欢的那辆车",
            "data_group_1": p1.coord.tolist(),
            "data_group_2": p2.coord.tolist(),
            "integer_value": num_question
        }
        # 向客户端发送数据
        await websocket.send(json.dumps(message))

        # 等待客户端输入
        try:
            # 接收来自客户端的消息
            response = await websocket.recv()

            # 解析 JSON 字符串
            data = json.loads(response)

            # 提取 "choice" 值
            if "choice" in data:
                choice = data["choice"]
                print(f"Received choice from client: {choice}")
                return choice
            else:
                print("No valid choice found in response")

        except json.JSONDecodeError:
            # 处理 JSON 解析错误
            print("Failed to decode JSON")
        except ValueError:
            # 处理无效输入
            print("Wrong Value")
        except websockets.ConnectionClosed:
            print("Client disconnected")