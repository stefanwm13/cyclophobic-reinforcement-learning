from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8, steps=10*8*8):
        super().__init__(
            grid_size=size,
            max_steps=steps
        )

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
        self.put_obj(Key('yellow'), 2, 3)
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

class DoorKeyEnv8x8Rand(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """

   
    def __init__(self):
        super().__init__(size=8, steps=10*8*8)

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
        super().__init__(size=11, steps=10*11*11)

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



class DKUnlockPickupRand(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """

   
    def __init__(self):
        super().__init__(size=8, steps=10*8*8)

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




class DoorKey2Env8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, steps=10*8*8)
        
    
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
        super().__init__(size=8, steps=10*8*8)
        
    
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



class DoorKeyYEnv8x8(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, steps=10*8*8)
        
    
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
        super().__init__(size=8, steps=10*8*8)
        
    
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
        super().__init__(size=8, steps=30*8*8)
        
    
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
        
        

class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)

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
    id='MiniGrid-DoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6'
)

register(
    id='MiniGrid-UnlockPickup-8x8-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickup'
)

register(
    id='MiniGrid-UnlockPickupRand-8x8-v0',
    entry_point='gym_minigrid.envs:DKUnlockPickupRand'
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
