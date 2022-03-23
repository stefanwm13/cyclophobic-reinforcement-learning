from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import random

class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        #Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        #Place the agent at a random position and orientation on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        #Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        #self.agent_pos =(1,1)
        #self.agent_dir = 2
        #self.put_obj(Key('red'), 2, 1)
        #self.put_obj(Key('yellow'), 2, 3)
        #self.put_obj(Door('yellow', is_locked=True), 4, 1)
        #self.put_obj(Wall(), 4, 2)
        #self.put_obj(Wall(), 4, 3)
        #self.put_obj(Wall(), 4, 4)
        #self.put_obj(Wall(), 4, 5)
        #self.put_obj(Wall(), 4, 6)
        #self.put_obj(Wall(), 4, 7)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        self.place_obj(
           obj=Key('yellow'),
           top=(0, 0),
           size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"


class DoorKeyEnv8x8Rand(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """


    def __init__(self):
        super().__init__(size=8)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"


class DoorKeyEnv11x11Rand(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """


    def __init__(self):
        super().__init__(size=11)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"




class Unlock5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5 )


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)



        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.agent_pos =(2,1)
        self.agent_dir = 1


        self.door1 = Door('green', is_locked=False)
        self.door2 = Door('red', is_locked=True)
        self.key = Key('red')
        self.ball = Ball('red')
        self.put_obj(self.door1, 1, 2)
        self.put_obj(self.door2, 3, 2)
        self.put_obj(self.key, 1, 3)
        self.put_obj(self.ball, 3, 3)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)

        self.mission = "use the key to open the door and then get to the goal"


    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.ball:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class Unlock7x7(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=7)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)



        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.agent_pos =(3,3)
        self.agent_dir = 3


        self.door1 = Door('green', is_locked=False)
        self.door2 = Door('green', is_locked=False)
        self.door3 = Door('green', is_locked=False)
        self.door4 = Door('green', is_locked=False)
        self.door5 = Door('red', is_locked=True)
        self.key = Key('red')
        self.ball = Ball('red')
        self.put_obj(self.door1, 1, 2)
        self.put_obj(self.door2, 2, 1)
        self.put_obj(self.door3, 2, 3)
        self.put_obj(self.door4, 4, 1)
        self.put_obj(self.door5, 4, 3)
        self.put_obj(self.key, 1, 1)
        self.put_obj(self.ball, 5, 3)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 5, 2)

        self.put_obj(Wall(), 1, 4)
        self.put_obj(Wall(), 2, 4)
        self.put_obj(Wall(), 3, 4)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 5, 4)
        self.mission = "use the key to open the door and then get to the goal"


    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.ball:
                reward = self._reward()
                done = True

        return obs, reward, done, info



class Unlock5x5Rand(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        colors = ['red', 'green', 'blue' ,'purple', 'yellow', 'grey', 'white']

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.agent_pos =(1,3)
        self.agent_dir = 2

        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        self.door = Door(random.choice(colors), is_locked=False)

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(self.door, splitIdx, doorIdx)


        self.mission = "use the key to open the door and then get to the goal"


    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.door.is_open:
            reward = self._reward()
            done = True

        return obs, reward, done, info



class DKUnlockPickupDoor(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=10)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        self.box = Box('red')
        self.put_obj(self.box, 8, 1)


        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 2, 6)
        self.put_obj(Door('yellow', is_locked=True), 3, 1)
        self.put_obj(Wall(), 3, 2)
        self.put_obj(Wall(), 3, 3)
        self.put_obj(Wall(), 3, 4)
        self.put_obj(Wall(), 3, 5)
        self.put_obj(Wall(), 3, 6)
        self.put_obj(Wall(), 3, 7)
        self.put_obj(Wall(), 3, 8)
        self.put_obj(Wall(), 3, 9)

        self.put_obj(Wall(), 6, 1)
        self.put_obj(Wall(), 6, 2)
        self.put_obj(Wall(), 6, 3)
        self.put_obj(Door('green', is_locked=False), 6, 4)
        self.put_obj(Wall(), 6, 5)
        self.put_obj(Wall(), 6, 6)
        self.put_obj(Wall(), 6, 7)
        self.put_obj(Wall(), 6, 8)
        self.put_obj(Wall(), 6, 9)




        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)


        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.box:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class DKUnlockPickupRandDoor(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """


    def __init__(self):
        super().__init__(size=10)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-4)
        self.grid.vert_wall(splitIdx, 0)

        # Create a second vertical splitting wall
        print(width)
        print(splitIdx)
        splitIdx2 = self._rand_int(splitIdx + 2, width-2)
        print(splitIdx2)
        self.grid.vert_wall(splitIdx2, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        doorIdx2 = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)
        self.put_obj(Door('green', is_locked=False), splitIdx2, doorIdx2)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        self.box = Box('red')
        self.place_obj(obj=self.box, top=(splitIdx2 + 1,1), size=(width - splitIdx2, height))

        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.box:
                print("CARRYING")
                reward = self._reward()
                done = True

        return obs, reward, done, info





class DKUnlockPickupRand(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """


    def __init__(self):
        super().__init__(size=8)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        self.box = Box('red')
        self.place_obj(obj=self.box, top=(splitIdx + 1,1), size=(width - splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.box:
                print("CARRYING")
                reward = self._reward()
                done = True

        return obs, reward, done, info


class RoomCorridor(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=14)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), 12, 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Wall(), 3, 1)
        #self.put_obj(Door('yellow', is_locked=True), 3, 1)
        self.put_obj(Wall(), 3, 3)
        self.put_obj(Wall(), 3, 4)
        self.put_obj(Wall(), 3, 5)
        self.put_obj(Wall(), 3, 6)
        self.put_obj(Wall(), 3, 7)
        self.put_obj(Wall(), 3, 8)
        self.put_obj(Wall(), 3, 9)
        self.put_obj(Wall(), 3, 10)
        self.put_obj(Wall(), 3, 11)
        self.put_obj(Wall(), 3, 12)
        self.put_obj(Wall(), 3, 13)

        #self.put_obj(Wall(), 1, 4)
        #self.put_obj(Wall(), 2, 4)

        self.put_obj(Wall(), 4, 1)
        self.put_obj(Wall(), 5, 1)
        self.put_obj(Wall(), 6, 1)
        self.put_obj(Wall(), 7, 1)
        self.put_obj(Wall(), 8, 1)
        self.put_obj(Wall(), 9, 1)
        self.put_obj(Wall(), 10, 1)
        self.put_obj(Wall(), 11, 1)
        self.put_obj(Wall(), 12, 1)

        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 5, 3)
        self.put_obj(Wall(), 6, 3)
        self.put_obj(Wall(), 7, 3)
        self.put_obj(Wall(), 8, 3)
        self.put_obj(Wall(), 9, 3)
        self.put_obj(Wall(), 10, 3)
        self.put_obj(Wall(), 11, 3)
        self.put_obj(Wall(), 12, 3)


        self.put_obj(Wall(), 1, 5)
        self.put_obj(Wall(), 1, 6)
        self.put_obj(Wall(), 1, 7)
        self.put_obj(Wall(), 1, 8)
        self.put_obj(Wall(), 1, 9)
        self.put_obj(Wall(), 1, 10)
        self.put_obj(Wall(), 1, 11)
        self.put_obj(Wall(), 1, 12)

        #self.put_obj(Wall(), 1, 3)
        #self.put_obj(Wall(), 1, 3)
        #self.put_obj(Wall(), 1, 3)
        #self.put_obj(Wall(), 1, 3)
        self.put_obj(Wall(), 2, 10)


        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"


class DoorKey2Env8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 3, 6)
        self.put_obj(Door('yellow', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"

class DKUnlockPickup(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        self.box = Box('red')
        self.put_obj(self.box, 6, 1)


        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 3, 6)
        self.put_obj(Door('yellow', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)



        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)


        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.box:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class DKUnlockPickup5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        self.box = Box('red')
        self.put_obj(self.box, 3, 2)


        # Setting for 8x8

        self.agent_pos =(1,3)
        self.agent_dir = 3

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)


        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)


        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.box:
                reward = self._reward()
                done = True

        return obs, reward, done, info



class DKUnlockPickup6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=7)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        self.box = Box('red')
        self.put_obj(self.box, 4, 4)


        # Setting for 8x8

        self.agent_pos =(1,5)
        self.agent_dir = 3

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 3, 1)
        self.put_obj(Wall(), 3, 2)
        self.put_obj(Wall(), 3, 3)
        self.put_obj(Wall(), 3, 4)
        self.put_obj(Wall(), 3, 5)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)


        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.box:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class DoorKeyCREnv8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        print(random.sample(list(COLORS), 1))

        self.put_obj(Key(random.sample(list(COLORS), 1)[0]), 3, 6)
        self.put_obj(Door('yellow', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"



class DoorKeyYEnv8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 3, 6)
        self.put_obj(Door('yellow', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"



class DoorKeyYEnv11x11(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=11)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 3, 6)
        self.put_obj(Door('yellow', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)
        self.put_obj(Wall(), 4, 8)
        self.put_obj(Wall(), 4, 9)
        self.put_obj(Wall(), 4, 10)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"


class DoorKeyYEnv11x11T(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=11, steps=10*11*11)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 4, 5)
        self.put_obj(Door('yellow', is_locked=True), 8, 1)
        self.put_obj(Wall(), 8, 2)
        self.put_obj(Wall(), 8, 3)
        self.put_obj(Wall(), 8, 4)
        self.put_obj(Wall(), 8, 5)
        self.put_obj(Wall(), 8, 6)
        self.put_obj(Wall(), 8, 7)
        self.put_obj(Wall(), 8, 8)
        self.put_obj(Wall(), 8, 9)
        self.put_obj(Wall(), 8, 10)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeyREnv8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('red'), 3, 6)
        self.put_obj(Door('red', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"


class DoorKey3Env8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)



        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width-2)
        #self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        #doorIdx = self._rand_int(1, width-2)
        #self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        # Setting for 5x5

        '''
        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)
        '''
        # Setting for 8x8

        self.agent_pos =(1,1)
        self.agent_dir = 2

        self.put_obj(Key('red'), 2, 1)
        self.put_obj(Key('yellow'), 3, 6)
        self.put_obj(Door('red', is_locked=True), 5, 3)
        self.put_obj(Door('yellow', is_locked=True), 4, 1)
        self.put_obj(Wall(), 4, 2)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)
        self.put_obj(Wall(), 6, 3)

        # Setting for 16x16

        #self.agent_pos =(1,14)
        #self.agent_dir = 2

        #self.put_obj(Key('yellow'), 5, 8)
        #self.put_obj(Door('yellow', is_locked=True), 7, 1)
        #self.put_obj(Wall(), 7, 2)
        #self.put_obj(Wall(), 7, 3)
        #self.put_obj(Wall(), 7, 4)
        #self.put_obj(Wall(), 7, 5)
        #self.put_obj(Wall(), 7, 6)
        #self.put_obj(Wall(), 7, 7)
        #self.put_obj(Wall(), 7, 8)
        #self.put_obj(Wall(), 7, 9)
        #self.put_obj(Wall(), 7, 10)
        #self.put_obj(Wall(), 7, 11)
        #self.put_obj(Wall(), 7, 12)
        #self.put_obj(Wall(), 7, 13)
        #self.put_obj(Wall(), 7, 14)
        #self.put_obj(Wall(), 7, 15)

        #self.place_obj(
        #    obj=Key('yellow'),
        #    top=(0, 0),
        #    size=(splitIdx, height)
        #)

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeyEnv5x5Y(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        self.agent_pos =(1,3)
        self.agent_dir = 2

        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('yellow', is_locked=True), 2, 1)
        self.put_obj(Wall(), 2, 2)
        self.put_obj(Wall(), 2, 3)

        self.mission = "use the key to open the door and then get to the goal"




class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class DoorKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5'
)

register(
    id='MiniGrid-DoorKeyY-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5Y'
)

register(
    id='MiniGrid-DoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6'
)

register(
    id='MiniGrid-UnlockPickup-8x8-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickup'
)

register(
    id='MiniGrid-UnlockPickupDoor-8x8-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickupDoor'
)

register(
    id='MiniGrid-RoomCorridor-8x8-v0',
    entry_point='gym_minigrid.envs:RoomCorridor'
)

register(
    id='MiniGrid-UnlockPickup-5x5-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickup5x5'
)

register(
    id='MiniGrid-UnlockPickup-6x6-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickup6x6'
)

register(
    id='MiniGrid-UnlockPickupRand-8x8-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickupRand'
)

register(
    id='MiniGrid-UnlockPickupRandDoor-8x8-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickupRandDoor'
)

register(
    id='MiniGrid-DoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv'
)

register(
    id='MiniGrid-DoorKey2-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKey2Env8x8'
)

register(
    id='MiniGrid-DoorKeyY-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyYEnv8x8'
)

register(
    id='MiniGrid-DoorKeyCR-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyCREnv8x8'
)

register(
    id='MiniGrid-Unlock-5x5-v0',
    entry_point='gym_minigrid.envs:Unlock5x5'
)

register(
    id='MiniGrid-Unlock-7x7-v0',
    entry_point='gym_minigrid.envs:Unlock7x7'
)

register(
    id='MiniGrid-UnlockRand-5x5-v0',
    entry_point='gym_minigrid.envs:Unlock5x5Rand'
)

register(
    id='MiniGrid-DoorKeyY-11x11-v0',
    entry_point='gym_minigrid.envs:DoorKeyYEnv11x11'
)

register(
    id='MiniGrid-DoorKeyY-11x11T-v0',
    entry_point='gym_minigrid.envs:DoorKeyYEnv11x11T'
)

register(
    id='MiniGrid-DoorKeyR-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyREnv8x8'
)

register(
    id='MiniGrid-DoorKey3-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKey3Env8x8'
)

register(
    id='MiniGrid-DoorKeyRand-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv8x8Rand'
)

register(
    id='MiniGrid-DoorKeyRand-11x11-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv11x11Rand'
)

register(
    id='MiniGrid-DoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv16x16'
)
