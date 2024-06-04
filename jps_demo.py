import time

import numpy as np
import matplotlib.pyplot as plt
import math
import heapq

show_animation = True

class Node:
    def __init__(self,pos,parent,g,h):
        self.pos = np.array(pos)
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g+h
        self.pos_set = (self.pos[0], self.pos[1])
    def get_direction(self):
        if self.parent is not None:
            direction = self.pos - self.parent.pos
            direction[direction > 0] = 1
            direction[direction < 0] = -1
        else:
            direction = np.array([0,0])
        return direction
def get_distance(node_1,node_2):
    dis = node_1.pos - node_2.pos
    return np.linalg.norm(dis)

def calc_distance(pos_1:np.array,pos_2:np.array)->np.array:
    dis = np.array(pos_1) - np.array(pos_2)
    return np.linalg.norm(dis)

class JPS:
    def __init__(self,ox,oy,resolution,rr):
        self.ox = ox
        self.oy = oy
        self.resolution = resolution
        self.rr = rr
        self.obstacle_map = None
        self.search_direction = [
            [1, 0], [0, 1], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]
        ]
        self.calc_obstacle_map(ox, oy)


    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)
    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    def check_node(self, cx,cy) -> bool:
        '''
        function: check the node is ok
        :param node: path node of map
        :return: the result of the node can be add to path
        '''
        # px = self.calc_grid_position(cx, self.min_x)
        # py = self.calc_grid_position(cy, self.min_y)
        px,py = cx,cy
        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False
        cix = int(self.calc_xy_index(cx,self.min_x))
        ciy = int(self.calc_xy_index(cy,self.min_x))
        # collision check
        if self.obstacle_map[cix][ciy]:
            return False

        return True
    def get_force_neighbor(self,current_node:Node) -> list:
        neighbors = []
        reso = self.resolution
        if current_node.parent == None:
            # 如果搜索节点式当前节点
            for dir in self.search_direction:
                nabor = [current_node.pos[0] + dir[0]*reso,
                        current_node.pos[1] + dir[1]*reso]
                if self.check_node(nabor[0],nabor[1]):
                    neighbors.append(dir)
        else:
            dir = current_node.get_direction()
            nabor = [current_node.pos[0] + dir[0],
                     current_node.pos[1] + dir[1]]
            if self.check_node(nabor[0],nabor[1]):
                neighbors.append(dir)
            # 如果是斜向移动的
            if dir[0] != 0 and dir[1] != 0:
                # 如果上方能走
                if self.check_node(current_node.pos[0] ,
                                   current_node.pos[1] + dir[1]*reso):

                    neighbors.append([0,dir[1]])
                # 如果右侧能走
                if self.check_node(current_node.pos[0] + dir[0]*reso,
                                   current_node.pos[1] ):
                    neighbors.append([dir[0],0])
                # 下侧不可走但右侧可走 拓展右下
                if (not self.check_node(current_node.pos[0] ,current_node.pos[1] - dir[1] * reso)
                    and self.check_node(current_node.pos[0]+ dir[0] * reso ,current_node.pos[1])) :

                # if (not self.check_node(current_node.pos[0] ,current_node.pos[1] + dir[0] * reso)
                #     and self.check_node(current_node.pos[0]+ dir[1] * reso ,current_node.pos[1])) :

                    neighbors.append([dir[0],-dir[1]])
                # 左不能走但上可走 拓展左上
                if (not self.check_node(current_node.pos[0] - dir[0] * reso,current_node.pos[1]) and
                    self.check_node(current_node.pos[0] ,current_node.pos[1] + dir[1] * reso)):
                    neighbors.append([-dir[0],dir[1]])
            else:
                # 如果是直线行走
                # 如果是垂直走，以向上走为例
                if dir[0] == 0:
                    # 如果右侧不能走，则拓展右上
                    if not self.check_node(current_node.pos[0] + reso,current_node.pos[1]):
                        neighbors.append([1,dir[1]])
                    # 如果左侧不能走，则拓展左上
                    if not self.check_node(current_node.pos[0] - reso,current_node.pos[1]):
                        neighbors.append([-1,dir[1]])
                # 如果是水平走
                else:
                    # 以右侧走为例
                    # 如果上侧不能走，则拓展右上
                    if not self.check_node(current_node.pos[0],current_node.pos[1] + reso):
                        neighbors.append([1,1])
                    # 如果下侧不能走则拓展右下
                    if not self.check_node(current_node.pos[0],current_node.pos[1] - reso):
                        neighbors.append([1,-1])

        return neighbors
    def jump_node(self,current:np.array,dir:np.array) -> bool:
        reso = self.resolution
        if current[0] == self.goal_node.pos[0] and current[1] == self.goal_node.pos[1]:
            # 如果当前节点到达目标节点
            return True
        if self.check_node(current[0],current[1]) == False:
            # 如果跳跃节点不可行
            return False
        if dir[0] != 0 and dir[1] != 0:
            # 如果是斜向前进
            # 如果右上可行且上不可行 或 左下可行且左不可行
            if ((self.check_node(current[0] + dir[0]*reso ,current[1] + reso) and not self.check_node(current[0],current[1]+reso)) or
                (self.check_node(current[0] + dir[0]*reso ,current[1] - reso) and not self.check_node(current[0],current[1]-reso))):
                return True
        else:
            # 如果是直线前进
            if dir[0] != 0:
                # 如果是水平方向前进
                '''
                * 1 0       0 0 0
                0 → 0       0 0 0
                * 1 0       0 0 0

                '''
                # 如果右上能走且右不能走或右下能走且下不能走
                if ((self.check_node(current[0] + dir[0]*reso, current[1] + reso) and not self.check_node(current[0], current[1] + reso)) or
                    (self.check_node(current[0] + dir[0]*reso, current[1] - reso) and not self.check_node(current[0], current[1] - reso))):
                    return True
            else:
                # 如果是垂直方向前进
                '''
                0 0 0
                1 ↓ 1
                0 0 0

                '''
                # 如果左下能走且左不能走或 右下能走且右不能走
                if ((self.check_node(current[0] - reso, current[1] + dir[1]*reso) and not self.check_node(current[0] - reso, current[1])) or
                    (self.check_node(current[0] + reso, current[1] + dir[1]*reso) and not self.check_node(current[0] + reso, current[1]))):
                    return True
        return False
    def jump_search(self,cur_Node:Node,search_direction:list):
        dir = np.array(search_direction)
        pos_cur = cur_Node.pos + dir * self.resolution
        if dir[0] == 0 or dir[1] == 0:
            # 如果是直线移动
            while self.check_node(pos_cur[0],pos_cur[1]):
                # 检查位置是否可行
                if self.jump_node(pos_cur,dir):
                    # 检查是否跳跃完成
                    return pos_cur
                else:
                    if show_animation:
                        plt.plot(pos_cur[0], pos_cur[1], "xc")
                        plt.pause(1e-5)
                    pos_cur += dir * self.resolution

        else:
            # 如果是斜向移动，则使用横纵向分别拓展寻找跳跃点
            horizontal = np.array([dir[0],0])
            vertical   = np.array([0,dir[1]])
            # 判断当前位置是否可行
            while self.check_node(pos_cur[0],pos_cur[1]):
                if self.jump_node(pos_cur,dir):
                    return pos_cur

                horizon_pos = pos_cur + self.resolution * horizontal

                # 判断横向是否可行
                while self.check_node(horizon_pos[0],horizon_pos[1]):
                    if self.jump_node(horizon_pos,horizontal):
                        return pos_cur
                    if show_animation:
                        plt.plot(horizon_pos[0], horizon_pos[1], "xc")
                        plt.pause(1e-5)
                    horizon_pos += self.resolution * horizontal

                vertical_pos = pos_cur + self.resolution * vertical
                # plt.plot(horizon_pos[0], horizon_pos[1], "xc")
                # plt.pause(0.001)
                # 判断垂向是否可行
                while self.check_node(vertical_pos[0],vertical_pos[1]):
                    if self.jump_node(vertical_pos,vertical):
                        return pos_cur
                    if show_animation :
                        plt.plot(vertical_pos[0], vertical_pos[1], "xc")
                        plt.pause(1e-5)
                    vertical_pos += self.resolution * vertical

                if not self.check_node(horizon_pos[0],horizon_pos[1]) and  self.check_node(vertical_pos[0],vertical_pos[1]):
                    break
                pos_cur += self.resolution*dir
        return None

    def search(self,start,goal):
        self.start_node = Node(start, None, 0 , calc_distance(start,goal))
        self.goal_node = Node(goal, None, 0,0)
        self.start_node.f = get_distance(self.start_node,self.goal_node)
        open_set = [self.start_node]
        heapq.heapify(open_set)
        close_set = set()
        while open_set:
            # 取出损失值最小的节点
            current_node = heapq.heappop(open_set)
            print(current_node.pos)
            if current_node.pos_set in close_set:
                continue
            close_set.add(current_node.pos_set)
            if current_node.pos_set == self.goal_node.pos_set:
                # 如果当前节点是目标节点，则结束
                self.goal_node = current_node
                break
            direction = self.get_force_neighbor(current_node)
            for dir in direction:
                jump_point = self.jump_search(current_node,dir)
                if jump_point is not None:
                    jp = (jump_point[0],jump_point[1])
                    # 如果跳跃点在close set中/ 被探索过
                    if jp in close_set:
                        continue
                    # 构建跳跃节点
                    next_node =  Node(jump_point,current_node,0,calc_distance(jump_point,goal))
                    plt.scatter(next_node.pos[0],next_node.pos[1])
                    heapq.heappush(open_set,next_node)

        if self.goal_node.parent is None:
            print("search failed")
        path = list()
        cur_node = self.goal_node
        while cur_node is not None:
            path.append(cur_node.pos)
            cur_node = cur_node.parent
        path = np.array(path)[::-1]
        plt.plot(path[:,0],path[:,1])
        plt.scatter(path[:,0],path[:,1])
        plt.show()
        return path




def main():

    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    # grid_size = 1.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        # plt.show()
    JPS_planner = JPS(ox, oy, grid_size, robot_radius)
    # JPS_planner = JPS(ox, oy, 1, 1)
    path = JPS_planner.search([sx, sy], [gx, gy])


if __name__ == '__main__':
    start = time.time()
    main()
    print("jps time use :" ,time.time() - start)