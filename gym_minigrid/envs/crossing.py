from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import random
import itertools as itt


class CrossingEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None, random_pos=False):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.random_pos = random_pos
        
        self.position_list = []
        self.position_list_r1 = []
        self.position_list_r2 = []
        self.position_list_r3 = []
        self.position_list_r4 = []
        random.seed(1337, version=2)
        for i in range(1,12):
            for j in range(1,12):
                #if (i == 6 and j == 1) or (i == 6 and j == 2) or (i == 6 and j == 4) or (i == 6 and j == 5) or (i == 6 and j == 6) or (i == 6 and j == 7) or (i == 6 and j == 8) \
                 #   or (i == 6 and j == 9) or (i == 6 and j == 10) or (i == 6 and j == 11) :
                if (i == 6 and j == 1) or (i == 6 and j == 2) or (i == 6 and j == 4) or (i == 6 and j == 5) or (i == 6 and j == 6) or (i == 6 and j == 7) or (i == 6 and j == 8) \
                    or (i == 6 and j == 9) or (i == 6 and j == 10) or (i == 6 and j == 11) or (i == 1 and j == 6) or (i == 2 and j == 6) or (i == 3 and j == 6) or (i == 4 and j == 6) or (i == 5 and j == 6) or (i == 7 and j == 7) or (i == 8 and j == 7) or (i == 10 and j == 7) or (i == 11 and j == 7) or (i == 11 and j == 11):
                    pass
                else:
                    self.position_list.append((i,j))
        
        self.sampled_list = []
        
        
        self.sampled_list_r1 = []
        self.sampled_list_r2 = []
        self.sampled_list_r3 = []
        self.sampled_list_r4 = []

        for i in range(1,6):
            for j in range(1,6):
                #if (i== 11 and j == 11):
                #    pass
                #else:
                #self.position_list_r1.append((6,3))
                self.position_list_r1.append((i,j))
            
        for i in range(7,12):
            for j in range(1,7):
                #if (i== 11 and j == 11):
                #    pass
                #else:
                #self.position_list_r2.append((9,7))
                self.position_list_r2.append((i,j))
                
        for i in range(1,6):
            for j in range(7,12):
                #if (i== 11 and j == 11):
                #    pass
                #else:
                #self.position_list_r3.append((2,6))
                self.position_list_r3.append((i,j))
                
        for i in range(7,12):
            for j in range(8,12):
                if (i== 11 and j == 11):
                    pass
                else:
                    #self.position_list_r4.append((6,10))
                    self.position_list_r4.append((i,j))
                    
        #self.sampled_list_r1 = random.sample(self.position_list_r1, 25)
        #self.sampled_list_r2 = random.sample(self.position_list_r2, 30)
        #self.sampled_list_r3 = random.sample(self.position_list_r3, 25)
        #self.sampled_list_r4 = random.sample(self.position_list_r4, 14)
        #self.sampled_list_r1.append((6,3))
        #self.sampled_list_r2.append((9,7))
        #self.sampled_list_r3.append((2,6))
        #self.sampled_list_r4.append((6,10))

        #self.sampled_list = self.sampled_list_r1 + self.sampled_list_r2 + self.sampled_list_r3 + self.sampled_list_r4
        
        self.sampled_list = random.sample(self.position_list, 50)
        self.diff_sampled_list = self.diff(self.position_list, self.sampled_list)
        #print(len(self.sampled_list))
        #print(len(self.diff_sampled_list))
        #print(self.diff_sampled_list)
        
        #print(self.sampled_list)
        #print(self.diff_sampled_list)
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )
        
    def diff(self, first, second):
        second = set(second)
        return [item for item in first if item not in second]

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        
        
        if self.random_pos:
            #print(len(self.position_list))
            #print("TEST")
            sampled_pos = random.sample(self.sampled_list, 1)
            #print(sampled_pos)
            self.agent_pos = sampled_pos[0]
            self.agent_dir = self._rand_int(0, 4)
        else:
            #print("NORMAL")
            #sampled_pos = random.sample(self.diff_sampled_list, 1)
            #print(self.diff_sampled_list)
            #self.agent_pos = sampled_pos[0]
            #self.agent_dir = self._rand_int(0, 4)
        
            self.agent_pos = (1,1)
            self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        
        
        # Place obstacles (lava or walls)
        '''
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)
        

        # Create openings
        
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(1,3))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(1, 3))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)
            if direction is h:
                self.grid.set(i, j, None)
            elif direction is v:
                self.grid.set(i, j, None)
            #self.grid.set(i+1, j+1, None)
        
        '''
        '''
        
        self.put_obj(self.obstacle_type(), 3, 1)
        self.put_obj(self.obstacle_type(), 3, 2)
        self.put_obj(self.obstacle_type(), 3, 4)
        self.put_obj(self.obstacle_type(), 3, 5)


        '''
        #self.put_obj(self.obstacle_type(), 4, 1)
        #self.put_obj(self.obstacle_type(), 4, 2)
        #self.put_obj(self.obstacle_type(), 4, 4)
        #self.put_obj(self.obstacle_type(), 4, 5)

        #self.put_obj(self.obstacle_type(), 6, 5)
        #self.put_obj(self.obstacle_type(), 6, 6)
        #self.put_obj(self.obstacle_type(), 6, 8)
        #self.put_obj(self.obstacle_type(), 6, 9)

        
        
        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

        
        self.put_obj(self.obstacle_type(), 6, 1)
        self.put_obj(self.obstacle_type(), 6, 2)
        
        self.put_obj(self.obstacle_type(), 6, 4)

        self.put_obj(self.obstacle_type(), 6, 5)
        self.put_obj(self.obstacle_type(), 6, 6)
        self.put_obj(self.obstacle_type(), 6, 7)
        #self.put_obj(self.obstacle_type(), 6, 8)
        #self.put_obj(self.obstacle_type(), 6, 9)
        #self.put_obj(self.obstacle_type(), 6, 11)
        
        #self.put_obj(self.obstacle_type(), 6, 10)

        
        self.put_obj(self.obstacle_type(), 1, 6)
        self.put_obj(self.obstacle_type(), 2, 6)
        self.put_obj(self.obstacle_type(), 3, 6)
        self.put_obj(self.obstacle_type(), 4, 6)
        self.put_obj(self.obstacle_type(), 5, 6)
        self.put_obj(self.obstacle_type(), 6, 6)
        self.put_obj(self.obstacle_type(), 7, 7)
        self.put_obj(self.obstacle_type(), 8, 7)
        self.put_obj(self.obstacle_type(), 10, 7)
        self.put_obj(self.obstacle_type(), 11, 7)
        self.put_obj(self.obstacle_type(), 12, 7)
        self.put_obj(self.obstacle_type(), 13, 7)
        self.put_obj(self.obstacle_type(), 14, 7)
        self.put_obj(self.obstacle_type(), 15, 7)

        
        
        self.put_obj(self.obstacle_type(), 1, 11)
        self.put_obj(self.obstacle_type(), 2, 11)
        self.put_obj(self.obstacle_type(), 3, 11)
        self.put_obj(self.obstacle_type(), 4, 11)
        self.put_obj(self.obstacle_type(), 5, 11)
        
        self.put_obj(self.obstacle_type(), 7, 11)
        self.put_obj(self.obstacle_type(), 8, 11)
        self.put_obj(self.obstacle_type(), 9, 11)
        self.put_obj(self.obstacle_type(), 10, 11)
        self.put_obj(self.obstacle_type(), 11, 11)
        self.put_obj(self.obstacle_type(), 12, 11)
        self.put_obj(self.obstacle_type(), 13, 11)
        self.put_obj(self.obstacle_type(), 14, 11)
        self.put_obj(self.obstacle_type(), 15, 11)
        
        '''
        self.put_obj(self.obstacle_type(), 8, 1)
        self.put_obj(self.obstacle_type(), 8, 3)
        self.put_obj(self.obstacle_type(), 8, 4)
        self.put_obj(self.obstacle_type(), 8, 5)

        self.put_obj(self.obstacle_type(), 8, 6)
        self.put_obj(self.obstacle_type(), 8, 7)
        self.put_obj(self.obstacle_type(), 8, 8)
        self.put_obj(self.obstacle_type(), 8, 9)
        self.put_obj(self.obstacle_type(), 2, 1)
        self.put_obj(self.obstacle_type(), 2, 2)
        self.put_obj(self.obstacle_type(), 2, 3)
        self.put_obj(self.obstacle_type(), 2, 4)
        self.put_obj(self.obstacle_type(), 2, 5)
        self.put_obj(self.obstacle_type(), 2, 7)
        self.put_obj(self.obstacle_type(), 2, 8)
        self.put_obj(self.obstacle_type(), 2, 9)
        '''
        


class LavaCrossingEnv(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=1)

class LavaCrossingS9N2Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=15, num_crossings=1)

class LavaCrossingS9N3Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=3)

class LavaCrossingS11N5Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=11, num_crossings=5)

register(
    id='MiniGrid-LavaCrossingS9N1-v0',
    entry_point='gym_minigrid.envs:LavaCrossingEnv'
)

register(
    id='MiniGrid-LavaCrossingS9N2-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS9N2Env'
)

register(
    id='MiniGrid-LavaCrossingS9N3-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS9N3Env'
)

register(
    id='MiniGrid-LavaCrossingS11N5-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS11N5Env'
)

class SimpleCrossingEnv(CrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)

class SimpleCrossingS9N2Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=17, num_crossings=1, obstacle_type=Wall)
        
class SimpleCrossingS9N2EnvRandom(CrossingEnv):
    def __init__(self):
        super().__init__(size=13, num_crossings=1, obstacle_type=Wall, random_pos=True)

class SimpleCrossingS9N3Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=3, obstacle_type=Wall)

class SimpleCrossingS11N5Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=11, num_crossings=5, obstacle_type=Wall)

register(
    id='MiniGrid-SimpleCrossingS9N1-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingEnv'
)

register(
    id='MiniGrid-MultiRoomS13N4-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N2Env'
)

register(
    id='MiniGrid-MultiRoomS13N4Random-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N2EnvRandom'
)

register(
    id='MiniGrid-SimpleCrossingS9N3-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N3Env'
)

register(
    id='MiniGrid-SimpleCrossingS11N5-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS11N5Env'
)
