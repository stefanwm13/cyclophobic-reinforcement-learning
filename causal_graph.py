"""
    causal_graph.py
        This file contains multiple classes sourrounding the building and representation of a causal graph.
"""
import networkx as nx
import matplotlib.pyplot as plt
import copy
import xxhash

from collections import Counter


"""
    This class represents a causal node which contains the state hash and the corresponding observation.
"""
class Node:
    def __init__(self, state_id, obs):
        self.state_id = state_id
        self.obs = obs
        
        self.next_state_ids = {}
        
    def add_next_state(self, state_node):
        self.next_state_ids[state_node.state_id] = state_node
        
        
        
        
"""
    This class represents a causal graph and contains methods to build and draw the causal graph.
"""
class CausalGraph:
    def __init__(self):
        self.node_dict = {}
        self.G = nx.Graph()

    
    def add_node(self, state_node):
        self.node_dict[state_node.state_id] = state_node
        self.G.add_node(state_node.state_id, image=state_node.obs)
    
    
    def add_next_state(self, state, next_state):
        state = self.get_node(state)
        next_state = self.get_node(next_state)
        
        if state == None or next_state == None:
            return False
        
        state.add_next_state(next_state)
        self.G.add_edge(state.state_id, next_state.state_id)
        
        return True
        
    
    def get_node(self, state_id):
        if state_id in self.node_dict:
            return self.node_dict[state_id]
        
        return None
    
    def get_connected_components(self, state):
        return nx.node_connected_component(self.G, state)
    
    def draw(self):
        pos = nx.spring_layout(self.G, scale=10)
        nx.draw(
            self.G,
            with_labels=True,
            pos=pos,
            node_size=500,
            node_color='r'
        )

        ax = plt.gca()
        fig = plt.gcf()
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        piesize=0.1 # this is the image size
        p2=piesize/2.0
        for n in self.G:
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-p2,ya-p2, piesize, piesize])
            a.set_aspect('equal')
            a.imshow(self.G.nodes[n]['image'])
            a.axis('off')
        ax.axis('off')
        
        nx.draw_networkx_edges(self.G,pos,ax=ax)
        plt.show()
