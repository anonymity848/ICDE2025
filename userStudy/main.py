import uuid

from structure.hyperplane import Hyperplane
from structure.point import Point
from structure.point_set import PointSet
import uh
from structure import constant
import time
import highRL
import lowRL
import time
import asyncio
import websockets
import json


async def user_study(websocket, session_id):
    dataset_name = 'audiSkyline'
    epsilon = 0.1
    trainning_size = 10000
    action_size = 5

    # dataset
    pset = PointSet(f'{dataset_name}.txt')
    dim = pset.points[0].dim
    for i in range(len(pset.points)):
        pset.points[i].id = i
    u = Point(dim)

    await uh.max_utility(pset, u, 2, epsilon, 1000, constant.SIMPlEX, constant.EXACT_BOUND, dataset_name, websocket, session_id)

    await highRL.highRL(pset, u, epsilon, dataset_name, False, trainning_size, action_size, websocket, session_id)

    await lowRL.lowRL(pset, u, epsilon, dataset_name, False, trainning_size, action_size, websocket, session_id)

    await uh.max_utility(pset, u, 2, epsilon, 1000, constant.RANDOM, constant.EXACT_BOUND, dataset_name, websocket,
                         session_id)

    message = {"message": "Thank you for your participation! \n 谢谢参与！"}
    await websocket.send(json.dumps(message))

    await websocket.close(code=1000, reason="Task completed")

# 每当有客户端连接时都会调用这个处理函数
async def handler(websocket):
    session_id = str(uuid.uuid4())  # 生成唯一标识符
    print(f"Client connected with session ID: {session_id}")
    try:
        await user_study(websocket, session_id)  # 启动客户端的输入处理
    except websockets.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error: {e}")


# 启动 WebSocket 服务器
async def main():
    async with websockets.serve(handler, "localhost", 8000):
        print("Server started at ws://localhost:8000")
        await asyncio.Future()  # 保持服务器运行

if __name__ == "__main__":
    asyncio.run(main())