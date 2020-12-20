from copy import deepcopy
from queue import PriorityQueue

from wumpus import Hunter
from .linear_space import SmartCoordinate, SmartVector

class HuntWumpusState(object):
    """
    Represent a state of the Hunt the Wumpus game with: 
    - agent_location: SmartCoordinate
            represents the current location of the agent in the world
    - agent_orientation: SmartVector
            represents the current orientation of the agent in the world 
    - is_arrow_available: bool
            represents the availability of the arrow to the agent
    - wumpus_locations: [SmartCoordinate]
            the list of possible wumpuses locations in the world
    - heuristic_cost: number
            the value of the heuristic associated to this state

    @static properties
    - safe_locations: [SmartCoordinate] 
            represents the list of safe locations where the agent can move
    - goal_locations: [SmartCoordinate]
            represents the list of locations the agent should reach 
    """
    
    safe_locations = []
    goal_locations = []

    def __init__(self, agent_location, 
                       agent_orientation, 
                       is_arrow_available=True,
                       wumpus_locations=[],
                       heuristic_cost=0):

        self.agent_location = agent_location
        self.agent_orientation = agent_orientation
        self.is_arrow_available = is_arrow_available
        self.wumpus_locations = wumpus_locations
        self.heuristic_cost = heuristic_cost

    def __eq__(self, other):
        return self.agent_location == other.agent_location and \
               self.agent_orientation == other.agent_orientation and \
               self.is_arrow_available == other.is_arrow_available and \
               self.wumpus_locations == other.wumpus_locations
               
    def __hash__(self):
        return hash((self.agent_location.x, self.agent_location.y, self.agent_orientation.x, 
                     self.agent_orientation.y, self.is_arrow_available, str(self.wumpus_locations)))
    
    def __str__(self):
        return f"HuntWumpusState: (agent_location = {self.agent_location}," \
               + f"\n\tagent_orientation = {self.agent_orientation}," \
               + f"\n\tis_arrow_available = {self.is_arrow_available}," \
               + f"\n\twumpus_locations = {self.wumpus_locations}," \
               + f"\n\theuristic_cost = {self.heuristic_cost}," \
               + f"\n\tsafe_locations = {HuntWumpusState.safe_locations})"\
               + f"\n\tgoal_locations = {HuntWumpusState.goal_locations})"

    def __repr__(self):
        return f"HuntWumpusState: (agent_location = {self.agent_location}," \
               + f"\n\tagent_orientation = {self.agent_orientation}," \
               + f"\n\tis_arrow_available = {self.is_arrow_available}," \
               + f"\n\twumpus_locations = {self.wumpus_locations}," \
               + f"\n\theuristic_cost = {self.heuristic_cost}," \
               + f"\n\tsafe_locations = {HuntWumpusState.safe_locations})"\
               + f"\n\tgoal_locations = {HuntWumpusState.goal_locations})"

    @staticmethod
    def setup_static_properties(safe_locations, goal_locations):
        HuntWumpusState.safe_locations = safe_locations
        HuntWumpusState.goal_locations = goal_locations

class HuntWumpusNode(object):
    """
    Represents a node of the problem with:
    - state: HuntWumpusState
            represents the state of the node
    - path_cost: number
            is the cost it takes to get to the node from the initial node applying 
            all previous_action defined in node parents
    - previous_action: Hunter.Action
            the action that was applied to the parent node to get to this node
    - parent: HuntWumpusNode
            parent node of the actual node
    """

    def __init__(self, state, path_cost=0, previous_action=None, parent=None):
        self.state = state
        self.path_cost = path_cost
        self.previous_action = previous_action
        self.parent = parent

    def __hash__(self):
        """
        Determines the uniqueness of an  HuntWumpusNode object. 
        Nodes should have same hash if they represent same state, meanwhile other attributes 
        can differ.
        """
        return self.state.__hash__()

    def __eq__(self, other):
        """
        It's been called every time there is the need to compare 2 objects of type 
        HuntWumpusNode. 
        It is used to prevent the search algorithm from exploring the nodes that have 
        already been visited. 
        (This avoids infinite loops while searching)
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        It's been called to define an order between 2 objects of type HuntWumpusNode
        in the PriorityQueue.
        Order is determined by the sum of path cost and heuristic cost. 
        If both nodes have the same (cost + heuristic) value, we defined the 
        following Breaking Ties:
            Level 1) if nodes have different heuristic_cost values, then we choose the 
                     one with lower one
            Level 2) if nodes have different orientations, then we defined the 
                     hierarchy N > E > W > S
            Level 3) we return the node with the lowest orientation.y
        """
        if self.get_cost_heuristic_sum() == other.get_cost_heuristic_sum():
            if self.state.heuristic_cost == other.state.heuristic_cost:
                if self.state.agent_orientation == other.state.agent_orientation:
                    return self.state.agent_orientation.y < other.state.agent_orientation.y 
                else:
                    if self.state.agent_orientation.y != other.state.agent_orientation.y:
                        return self.state.agent_orientation.y > other.state.agent_orientation.y  
                    else:
                        self.state.agent_orientation.x > other.state.agent_orientation.x
            else:
                return self.state.heuristic_cost < other.state.heuristic_cost
        else:
            return self.get_cost_heuristic_sum() <= other.get_cost_heuristic_sum()

    def __str__(self):
        return f"HuntWumpusNode: (id = {id(self)}," \
               + f"\n\tstate = {str(self.state)}," \
               + f"\n\tparent = {id(self.parent)}," \
               + f"\n\tprevious_actions = {self.unwrap_previous_actions()}," \
               + f"\n\tpath_cost = {self.path_cost}," \
               + f"\n\tvalue_in_priority_queue = {self.get_cost_heuristic_sum()}," \
               + f"\n\tparent_heuristic - current_heuristic: " \
               + f"{self.parent.state.heuristic_cost - self.state.heuristic_cost if self.parent is not None else 0} " \
               + f"<= {self.previous_action}"

    def __repr__(self):
        return f"HuntWumpusNode: (id = {id(self)}," \
               + f"\n\tstate = {str(self.state)}," \
               + f"\n\tparent = {id(self.parent)}," \
               + f"\n\tprevious_actions = {self.unwrap_previous_actions()}," \
               + f"\n\tpath_cost = {self.path_cost}," \
               + f"\n\tvalue_in_priority_queue = {self.get_cost_heuristic_sum()}," \
               + f"\n\tparent_heuristic - current_heuristic: " \
               +f"{self.parent.state.heuristic_cost - self.state.heuristic_cost if self.parent is not None else 0} " \
               + f"<= {self.previous_action}"

    def get_cost_heuristic_sum(self):
        """
        This func is used to calculate the priority of nodes in the Priority Queue used by the 
        A* algorithm, lower values will result in a high priority
        """
        return self.path_cost + self.state.heuristic_cost

    def unwrap_previous_actions(self):
        """
        returns all actions performed from the initial_node to the given node
        """
        if self.parent is None:
            return []
        
        return self.parent.unwrap_previous_actions() + [self.previous_action]

class HuntWumpusProblem(object):
    """
    Is the formal representation of the hunt the wumpus problem in general:
    - initial_state: HuntWumpusState
            the initial state of the problem
    - possible_actions: [Hunter.Actions]
            is the list of possible actions that can be performed by the agent 
    - is_goal_state: (HuntWumpusState) -> Bool
            is the function that check whether the current search has reached the goal
    - heuristic_func: (HuntWumpusNode) -> number
            is a function that calculates the heuristic of the current node against 
            the goal of the problem
    - action_costs: { Hunter.Actions: lambda (state, action, next_state) -> number }
            is a mapping between each possible action available to the agent and the 
            associated function which can be used to calculate its cost
    """
  
    def __init__(self, safe_locations, goal_locations, possible_actions, is_goal_state, heuristic_func, 
                 agent_location, agent_orientation, is_arrow_available=False, wumpus_locations=set()):

        self.initial_state = HuntWumpusState(agent_location=agent_location, 
                                             agent_orientation=agent_orientation, 
                                             is_arrow_available=is_arrow_available, 
                                             wumpus_locations=wumpus_locations)

        HuntWumpusState.setup_static_properties(safe_locations=safe_locations, goal_locations=goal_locations)
        
        self.possible_actions = possible_actions
        self.is_goal_state = is_goal_state
        self.heuristic_func = heuristic_func
        self.initial_state.heuristic_cost = self.heuristic_func(self.initial_state)

        # Action costs:
        # Shooting (using the arrow) -> 10 (otherwise 1)
        # All other -> 1
        self.action_costs = {
            Hunter.Actions.LEFT: lambda state, action, next_state: 1,
            Hunter.Actions.RIGHT: lambda state, action, next_state: 1,
            Hunter.Actions.MOVE: lambda state, action, next_state: 1,
            Hunter.Actions.SHOOT: lambda state, action, next_state: 10 if (state.is_arrow_available == True) 
                                                                          and (next_state.is_arrow_available == False)
                                                                    else 1
        }
        
    def is_legal(self, location, *, for_state):
        """
        returns a boolean indicating if the given location is one of the safe locations defined by the problem
        """
        state = for_state

        if state.wumpus_locations:
            return location in HuntWumpusState.safe_locations and not location in state.wumpus_locations
        else:
            return location in HuntWumpusState.safe_locations

    def get_available_actions_for(self, state):
        return self.possible_actions

    # effective actions (ones that do change the state of the problem)
    def get_effective_actions_for(self, state):
        """
        filters out all actions (from all the available ones) that do not change the state of 
        the world (wasteful actions)
        """
        available_actions = self.get_available_actions_for(state)

        # func definitions for switch statement that is used in next for loop
        def is_MOVE_effective_for(state):
            new_location = (state.agent_location + state.agent_orientation)
            return self.is_legal(new_location, for_state=state)
            
        def is_SHOOT_effective_for(state):
            return state.is_arrow_available

        switcher = {
            Hunter.Actions.MOVE: is_MOVE_effective_for,
            Hunter.Actions.SHOOT: is_SHOOT_effective_for
        }

        effective_actions = []

        for action in available_actions:
            if action == Hunter.Actions.LEFT or action == Hunter.Actions.RIGHT:
                effective_actions.append(action)
                continue

            is_action_effective_for = switcher.get(action, lambda x: False)
            if is_action_effective_for(state):
                effective_actions.append(action)
        
        return effective_actions

    def get_best_actions_for(self, state):
        """
        calculate the best rotation actions for the current state, improving the efficiency 
        of the rotation of the agent
        """

        effective_actions = set(self.get_effective_actions_for(state))
        useless_actions = set()

        # no shoot if there is no wumpus to kill
        if state.agent_location + state.agent_orientation not in state.wumpus_locations:
            useless_actions.add(Hunter.Actions.SHOOT)

        # no move into a pit
        if state.agent_location + state.agent_orientation not in HuntWumpusState.safe_locations:
            useless_actions.add(Hunter.Actions.MOVE)
        
        # best rotation moves to get around obstacles
        agent_orientation = state.agent_orientation
        perpendicular_orientation = state.agent_orientation.get_perpendicular_vector_clockwise()

        east_location = state.agent_location + perpendicular_orientation
        south_location = state.agent_location - agent_orientation
        west_location = state.agent_location - perpendicular_orientation

        if (not self.is_legal(west_location, for_state=state) 
           or west_location not in HuntWumpusState.safe_locations): # WEST is a block
            if (not self.is_legal(east_location, for_state=state) 
               or east_location not in HuntWumpusState.safe_locations): # EAST is a block
                if (not self.is_legal(south_location, for_state=state) 
                   or south_location not in HuntWumpusState.safe_locations): # SOUTH is a block
                    useless_actions = useless_actions.union(set([Hunter.Actions.RIGHT, Hunter.Actions.LEFT]))
                else: # SOUTH not a block
                    useless_actions = useless_actions.union(set([Hunter.Actions.LEFT]))
            else: # EAST not a block
                useless_actions = useless_actions.union(set([Hunter.Actions.LEFT]))
        else: #WEST not a block
            if (not self.is_legal(east_location, for_state=state) 
                or east_location not in HuntWumpusState.safe_locations): # EAST is a block
                useless_actions = useless_actions.union(set([Hunter.Actions.RIGHT]))

        return effective_actions - useless_actions

    def get_successor_state_from(self, state, *, with_action):
        """
        returns the successor HuntWumpusState resulting from applying the given action
        on the given state
        """
        action = with_action

        if action not in self.get_best_actions_for(state):
            return deepcopy(state)

        def get_LEFT_successor_from(state):
            return HuntWumpusState(state.agent_location, 
                                   -state.agent_orientation.get_perpendicular_vector_clockwise(),
                                   state.is_arrow_available, 
                                   state.wumpus_locations)

        def get_RIGHT_successor_from(state):
            return HuntWumpusState(state.agent_location, 
                                   state.agent_orientation.get_perpendicular_vector_clockwise(),
                                   state.is_arrow_available, 
                                   state.wumpus_locations)

        def get_MOVE_successor_from(state):
            new_location = (state.agent_location + state.agent_orientation)
            if not(self.is_legal(new_location, for_state=state)):
                new_location = state.agent_location

            return HuntWumpusState(new_location,
                                   state.agent_orientation, 
                                   state.is_arrow_available, 
                                   state.wumpus_locations)

        def get_SHOOT_successor_from(state):
            remaining_wumpus = list(filter(lambda element: element != (state.agent_location + state.agent_orientation), 
                                           state.wumpus_locations))
            return HuntWumpusState(state.agent_location, 
                                   state.agent_orientation,
                                   False, 
                                   remaining_wumpus)

        switcher = {
            Hunter.Actions.LEFT: get_LEFT_successor_from,
            Hunter.Actions.RIGHT: get_RIGHT_successor_from,
            Hunter.Actions.MOVE: get_MOVE_successor_from,
            Hunter.Actions.SHOOT: get_SHOOT_successor_from
        }

        get_successor_state_from = switcher.get(action, lambda x: deepcopy(state))
        successor = get_successor_state_from(state)
        successor.heuristic_cost = self.heuristic_func(successor)
        return successor

    def get_child_from(self, node, *, with_action):
        """
        returns the child HuntWumpusNode resulting from applying the given action on 
        the given node
        """   
        action = with_action

        next_state = self.get_successor_state_from(node.state, with_action=action)

        get_action_cost_from = self.action_costs.get(action, lambda x, y, z: 1)
        action_cost = get_action_cost_from(node.state, action, next_state)

        return HuntWumpusNode(next_state, node.path_cost + action_cost, action, node)

    def unwrap_solution(self, node):
        """
        returns all actions performed from the initial_node to the given node
        """
        if node.parent is None:
            return []
        
        return self.unwrap_solution(node.parent) + [node.previous_action]