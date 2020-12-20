import itertools
import math

from typing import NamedTuple, Iterable
from copy import deepcopy
from queue import PriorityQueue

from wumpus import Hunter
from .linear_space import SmartCoordinate, SmartVector
from .offline_search_model import HuntWumpusProblem
from .offline_search_model import HuntWumpusNode, HuntWumpusState

from z3 import *

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from .utility import get_pretty_log_representation_of, get_final_location
from .utility import manhattan_distance_between, smart_manhattan_distance

PIT_PROBABILITY = 0.2
MIN_SAFE_PROBABILITY_TRESHOLD = 0.76
IS_SQUARE_SIZE = True

# DISCLAIMER:
# Some empirical tests have shown that implementing a SAT solver to detect
# the locations where the pits must be in improve the efficiency of the online search,
# so we added a SAT solver to the KnwowledgeBase class.

class KnowledgeBase(object):
    """
    This class defines the knowledge base of the online search.
    It acquires more and more information as the search progresses and
    can derive essential data.
    It uses a SAT solver and a bayesian networtk to make some deeper inferences.
    """

    def __init__(self):

        self.world_size_width = 1
        self.world_size_height = 1
        self.has_agent_bumped_east = False
        self.has_agent_bumped_north = False

        self.agent_location = SmartCoordinate(0, 0)
        self.agent_orientation = SmartVector(0, 1)
        self.is_arrow_available = True

        self.exit_locations = set([SmartCoordinate(0, 0)])
        self.visited_locations = set()
        self.fringe_locations = set([SmartCoordinate(0, 0)])
        
        # calculated based on percepts
        self.no_pit_locations = set()
        self.no_wumpus_locations = set()

        self.breeze_locations = set()
        self.no_breeze_locations = set()

        self.possible_wumpus_locations = set()
        self.known_wumpus_location = None
        self.is_wumpus_alive = True

        self.pit_SAT_solver = Solver()

        # calculated with the SAT solver
        self.pit_locations = set()

    def __str__(self):
        return f"KnowledgeBase: (World Size: " \
                + f"width = {self.world_size_width}, " \
                + f"height = {self.world_size_height}," \
                + f"\n\thas_agent_bumped: " \
                + f"east = {self.has_agent_bumped_east}, " \
                + f"north = {self.has_agent_bumped_north}," \
                + f"\n\tagent_location = {self.agent_location}," \
                + f"\n\tagent_orientation = {self.agent_orientation}," \
                + f"\n\texit_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.exit_locations)}," \
                + f"\n\tvisited_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.visited_locations)}," \
                + f"\n\tfringe_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.fringe_locations)}," \
                + f"\n\tno_pit_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.no_pit_locations)}," \
                + f"\n\tno_wumpus_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.no_wumpus_locations)}," \
                + f"\n\tbreeze_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.breeze_locations)}," \
                + f"\n\tno_breeze_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.no_breeze_locations)}," \
                + f"\n\tpit_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.pit_locations)}," \
                + f"\n\tpossible_wumpus_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.possible_wumpus_locations)}," \
                + f"\n\tknown_wumpus_location = " \
                + f"{self.known_wumpus_location}," \
                + f"\n\tis_wumpus_alive = " \
                + f"{self.is_wumpus_alive}," \
                + f"\n\tsafe_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.get_safe_locations())}," \
                + f"\n\tsafe_unvisited_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.get_safe_unvisited_locations())})"

    def __repr__(self):
        return f"KnowledgeBase: (World Size: " \
                + f"width = {self.world_size_width}, " \
                + f"height = {self.world_size_height}," \
                + f"\n\thas_agent_bumped: " \
                + f"east = {self.has_agent_bumped_east}, " \
                + f"north = {self.has_agent_bumped_north}," \
                + f"\n\tagent_location = {self.agent_location}," \
                + f"\n\tagent_orientation = {self.agent_orientation}," \
                + f"\n\texit_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.exit_locations)}," \
                + f"\n\tvisited_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.visited_locations)}," \
                + f"\n\tfringe_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.fringe_locations)}," \
                + f"\n\tno_pit_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.no_pit_locations)}," \
                + f"\n\tno_wumpus_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.no_wumpus_locations)}," \
                + f"\n\tbreeze_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.breeze_locations)}," \
                + f"\n\tno_breeze_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.no_breeze_locations)}," \
                + f"\n\tpit_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.pit_locations)}," \
                + f"\n\tpossible_wumpus_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.possible_wumpus_locations)}," \
                + f"\n\tknown_wumpus_location = " \
                + f"{self.known_wumpus_location}," \
                + f"\n\tis_wumpus_alive = " \
                + f"{self.is_wumpus_alive}," \
                + f"\n\tsafe_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.get_safe_locations())}," \
                + f"\n\tsafe_unvisited_locations = " \
                + f"{get_pretty_log_representation_of(locations=self.get_safe_unvisited_locations())})"

    def get_adjacent_locations_from(self, *, location):
        """
        calculates each ortogonally adjacent location from the given one and returns the list of them
        """
        orientations = [SmartVector(element[0], element[1]) for element in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
        return set([location + orientation for orientation in orientations])

    def filter_locations_from_unknown_world(self, *, from_locations):
        """
        filters out any location outside of the current world size conception of the agent at the current time
        """
        max_east_range_border = self.world_size_width if self.has_agent_bumped_east else self.world_size_width + 1
        max_north_range_border = self.world_size_height if self.has_agent_bumped_north else self.world_size_height + 1
        inner_locations = set(filter(lambda location: location.x in range(0, max_east_range_border) \
                                                      and location.y in range(0, max_north_range_border), 
                                                      from_locations))
    
        return inner_locations

    def update_with(self, *, previous_agent_action, current_percept):
        """
        Updates the knowledge base's information with the current percept 
        """

        if previous_agent_action is not None:
            if previous_agent_action == Hunter.Actions.MOVE and not current_percept.bump:
                self.agent_location = self.agent_location + self.agent_orientation
            
            elif previous_agent_action == Hunter.Actions.RIGHT:
                self.agent_orientation = self.agent_orientation.get_perpendicular_vector_clockwise() 
            elif previous_agent_action == Hunter.Actions.LEFT:
                self.agent_orientation = -self.agent_orientation.get_perpendicular_vector_clockwise()
            
            elif previous_agent_action == Hunter.Actions.SHOOT:
                self.is_arrow_available = False

                if current_percept.scream:
                    self.update_wumpus(scream=True)
                else:
                    new_no_wumpus_locations = set()             
                    if self.agent_orientation == SmartVector.get_north():
                        new_no_wumpus_locations = set(filter(lambda x: x.x == self.agent_location.x \
                                                                       and x.y > self.agent_location.y, 
                                                                       self.possible_wumpus_locations))
                    if self.agent_orientation == SmartVector.get_east():
                        new_no_wumpus_locations = set(filter(lambda x: x.y == self.agent_location.y \
                                                                       and x.x > self.agent_location.x, 
                                                                       self.possible_wumpus_locations))
                    if self.agent_orientation == SmartVector.get_south():
                        new_no_wumpus_locations = set(filter(lambda x: x.x == self.agent_location.x \
                                                                       and x.y < self.agent_location.y, 
                                                                       self.possible_wumpus_locations))
                    if self.agent_orientation == SmartVector.get_west():
                        new_no_wumpus_locations = set(filter(lambda x: x.y == self.agent_location.y \
                                                                       and x.x < self.agent_location.x, 
                                                                       self.possible_wumpus_locations))

                    self.update_wumpus(with_no_wumpus_locations=new_no_wumpus_locations)

        if self.agent_location in self.visited_locations:
            if current_percept.bump:
                if IS_SQUARE_SIZE:
                    self.has_agent_bumped_north = True
                    self.has_agent_bumped_east = True
                else:
                    if self.agent_orientation == SmartVector(0, 1):
                        self.has_agent_bumped_north = True

                    if self.agent_orientation == SmartVector(1, 0):
                        self.has_agent_bumped_east = True
                    
                self.fringe_locations = self.filter_locations_from_unknown_world(from_locations=self.fringe_locations)

                self.no_pit_locations = self.filter_locations_from_unknown_world(from_locations=self.no_pit_locations)
                
                self.update_wumpus()

            return

        if IS_SQUARE_SIZE and (self.agent_location.x not in range(0, self.world_size_width) \
                          or self.agent_location.y not in range(0, self.world_size_height)) \
                          and not self.has_agent_bumped_east:
            self.world_size_width += 1
            self.world_size_height += 1
        else:
            if self.agent_location.x not in range(0, self.world_size_width) and not self.has_agent_bumped_east:
                self.world_size_width += 1

            if self.agent_location.y not in range(0, self.world_size_height) and not self.has_agent_bumped_north:
                self.world_size_height += 1

        self.no_pit_locations.add(self.agent_location)

        self.no_wumpus_locations.add(self.agent_location)

        self.visited_locations.add(self.agent_location)

        neighbour_locations = self.get_adjacent_locations_from(location=self.agent_location)
        inner_neighbour_locations = self.filter_locations_from_unknown_world(from_locations=neighbour_locations)
        
        not_visited_neighbour_locations = set(filter(lambda x: x not in self.visited_locations, neighbour_locations))
        inner_not_visited_neighbour_locations = self.filter_locations_from_unknown_world(from_locations=not_visited_neighbour_locations)
            
        self.fringe_locations.remove(self.agent_location)
        self.fringe_locations = self.fringe_locations.union(inner_not_visited_neighbour_locations)

        if current_percept.breeze:
            self.breeze_locations.add(self.agent_location)
            
            pit_inner_neighbours_smt_bool_variables = self.get_smt_bool_variables_from(locations=inner_neighbour_locations, with_prefix="pit")
            self.pit_SAT_solver.add(AtLeast(pit_inner_neighbours_smt_bool_variables + [1]))
            self.pit_SAT_solver.add(AtMost(pit_inner_neighbours_smt_bool_variables + [len(pit_inner_neighbours_smt_bool_variables)]))

            no_pit_smt_bool_variable = Bool(f'pit_{self.agent_location.x}_{self.agent_location.y}')
            self.pit_SAT_solver.add(no_pit_smt_bool_variable == False)
        else:
            self.no_breeze_locations.add(self.agent_location)

            pit_inner_neighbours_smt_bool_variables = self.get_smt_bool_variables_from(locations=inner_neighbour_locations, with_prefix="pit")
            self.pit_SAT_solver.add(AtMost(pit_inner_neighbours_smt_bool_variables + [0]))

            no_pit_smt_bool_variables = self.get_smt_bool_variables_from(locations=inner_not_visited_neighbour_locations.union(set([self.agent_location])), with_prefix="pit")
            self.add_assertions(for_variables=no_pit_smt_bool_variables, with_value=False, to_solver=self.pit_SAT_solver)
            
            self.no_pit_locations = self.no_pit_locations.union(inner_not_visited_neighbour_locations)

        if current_percept.stench:
            self.update_wumpus(with_possible_wumpus_locations=inner_not_visited_neighbour_locations)
        else:
            self.update_wumpus(with_no_wumpus_locations=inner_not_visited_neighbour_locations)

    def update_wumpus(self, *, with_possible_wumpus_locations=set(), with_no_wumpus_locations=set(), scream=False):
        """
        updates all relevant information discovered about the wumpus
        """
        possible_wumpus_locations = with_possible_wumpus_locations
        no_wumpus_locations = with_no_wumpus_locations

        self.no_wumpus_locations = self.no_wumpus_locations.union(no_wumpus_locations)
        
        self.no_wumpus_locations = self.filter_locations_from_unknown_world(from_locations=self.no_wumpus_locations)
        
        self.possible_wumpus_locations = self.filter_locations_from_unknown_world(from_locations=self.possible_wumpus_locations)
        
        if not self.is_wumpus_alive:
           return

        if scream:
            self.no_wumpus_locations = self.no_wumpus_locations.union(self.possible_wumpus_locations)

            self.is_wumpus_alive = False
            self.known_wumpus_location = None
            self.possible_wumpus_locations = set()
            return

        if self.known_wumpus_location is None and self.is_wumpus_alive:
            if possible_wumpus_locations:
                if not self.possible_wumpus_locations:
                    self.possible_wumpus_locations = possible_wumpus_locations
                else:
                    self.possible_wumpus_locations = self.possible_wumpus_locations.intersection(possible_wumpus_locations)

            self.possible_wumpus_locations = self.possible_wumpus_locations.difference(self.no_wumpus_locations)
            
            if len(self.possible_wumpus_locations) == 1:
                self.known_wumpus_location = list(self.possible_wumpus_locations)[0]
                self.no_wumpus_locations = self.no_wumpus_locations.union(possible_wumpus_locations.difference(set([self.known_wumpus_location])))
        
        if self.known_wumpus_location and self.is_wumpus_alive:
            self.no_wumpus_locations = self.no_wumpus_locations.union(possible_wumpus_locations.difference(set([self.known_wumpus_location])))

    def get_safe_locations(self):
        """
        returns the full list of safe locations that can be inferred from the knowledge base
        """
        if not self.is_wumpus_alive:
            return self.no_pit_locations
        else:
            if self.possible_wumpus_locations:
                return self.no_pit_locations.difference(self.possible_wumpus_locations)
            else:
                return self.no_pit_locations.intersection(self.no_wumpus_locations)

    def get_safe_unvisited_locations(self):
        """
        returns the full list of safe unvisited locations that can be inferred from the knowledge base
        """
        safe_locations = self.get_safe_locations()
        return safe_locations.difference(self.visited_locations)

    def get_safe_from_pit_but_possible_wumpus_locations(self):
        return self.no_pit_locations

    def get_smt_bool_variables_from(self, *, locations, with_prefix):
        """
        helper function to easily create a list of bool smt variables with a string pattern
        """
        prefix = with_prefix

        return [Bool("_".join([prefix, str(location.x), str(location.y)])) for location in locations]

    def add_assertions(self, *, for_variables, with_value, to_solver):
        """
        helper function to make an assertion that assign a list of variables to the same value 
        """
        variables = for_variables
        value = with_value
        solver = to_solver

        for variable in variables:
            solver.add(variable == value)

    def get_bool_value_if_certain(self, *, of_bool_variable, in_solver):
        """
        checks whether the assertion is consistent in ALL models generated by the solver
        """
        bool_variable = of_bool_variable
        solver = in_solver

        result1 = solver.check(bool_variable == True)
        result2 = solver.check(bool_variable == False)

        if result1 == CheckSatResult(Z3_L_TRUE) and result2 == CheckSatResult(Z3_L_FALSE):
            return True

        if result1 == CheckSatResult(Z3_L_FALSE) and result2 == CheckSatResult(Z3_L_TRUE):
            return False

        return None

    def update_pit_locations(self):
        """
        update the pits locations by running the solver and trying to infer new information
        """
        for location in self.fringe_locations.difference(self.no_pit_locations).difference(self.pit_locations):
            is_pit_location = self.get_bool_value_if_certain(of_bool_variable=Bool(f'pit_{location.x}_{location.y}'), in_solver=self.pit_SAT_solver)
            if is_pit_location is not None and is_pit_location:
                self.pit_locations.add(location)
                self.pit_SAT_solver.add(Bool(f'pit_{location.x}_{location.y}') == True)

    def make_bayesian_proposition(self, *, with_name, and_values, and_probabilities):
        """
        helper method to define a TabularCPD easily
        """
        name = with_name
        values = and_values
        probabilities = and_probabilities
        
        return TabularCPD(variable=name, variable_card=len(values), 
                          values=[[probability] for probability in probabilities], state_names={name: values})

    def make_evidence_bool_bayesian_proposition(self, *, with_name, and_values, and_evidence, with_bool_function):
        """
        helper method to make an evidence bayesian proposition easily
        """
        name = with_name
        values = and_values
        evidence = and_evidence
        bool_function = with_bool_function

        bool_function = eval(bool_function)
        
        def cpd_values(bool_fn, arity):
            def bool_to_prob(*args) -> bool:
                return (1.0, 0.0) if bool_fn(*args) else (0.0, 1.0)
    
            return tuple(zip(*[bool_to_prob(*ps) for ps in itertools.product((True, False), repeat=arity)]))

        return TabularCPD(
            variable=name, variable_card=2,
            values=cpd_values(bool_function, len(evidence)),
            evidence=evidence, evidence_card=[2] * len(evidence),
            state_names={n: values for n in [name] + evidence}
        )

    def make_bayesian_bool_propositions_dict(self, *, for_locations, with_name_prefix, and_probabilities):
        """
        generates the bayesian bool propositions dict
        """
        locations = for_locations
        name_prefix = with_name_prefix
        probabilities = and_probabilities

        cpds = {}
        for location in locations:
            cpds[SmartCoordinate(location.x, location.y)] = self.make_bayesian_proposition(with_name=f'{name_prefix}{location.x}{location.y}', 
                                                                                           and_values=[True, False], 
                                                                                           and_probabilities=probabilities)

        return cpds

    def get_bayesian_model(self):
        """
        make a bayesian model based on the information known by the knowledge base
        """
        global PIT_PROBABILITY

        pit_cpds_dict = {}
        pit_cpds_dict.update(self.make_bayesian_bool_propositions_dict(for_locations=self.fringe_locations.difference(self.no_pit_locations).difference(self.pit_locations),
                                                                                                                    with_name_prefix='P', 
                                                                                                                    and_probabilities=[PIT_PROBABILITY, 1 - PIT_PROBABILITY]))
        pit_cpds_dict.update(self.make_bayesian_bool_propositions_dict(for_locations=self.no_pit_locations, 
                                                                                                                    with_name_prefix='P', 
                                                                                                                    and_probabilities=[0, 1]))
        pit_cpds_dict.update(self.make_bayesian_bool_propositions_dict(for_locations=self.pit_locations,
                                                                                                                    with_name_prefix='P', 
                                                                                                                    and_probabilities=[1, 0]))


        breeze_cpds = {}
        for breeze_location in self.breeze_locations.union(self.no_breeze_locations):
            evidence = [f'P{inner_neighbour_location.x}{inner_neighbour_location.y}' for inner_neighbour_location in self.filter_locations_from_unknown_world(from_locations=self.get_adjacent_locations_from(location=breeze_location))]
            breeze_cpds[(breeze_location.x, breeze_location.y)] = self.make_evidence_bool_bayesian_proposition(with_name=f"B{breeze_location.x}{breeze_location.y}", 
                                                                                                                                                                                                and_values=[True, False], 
                                                                                                                                                                                                and_evidence=evidence, 
                                                                                                                                                                                                with_bool_function=f'lambda {", ".join(evidence)}: ({" or ".join(evidence)})')

        exp_model = BayesianModel()

        for breeze_cpd in breeze_cpds.values():
            for evidence in breeze_cpd.get_evidence():
                exp_model.add_edge(evidence, breeze_cpd.variable)

        exp_model.add_nodes_from([pit_cpd.variable for pit_cpd in pit_cpds_dict.values()])
        exp_model.add_cpds(*breeze_cpds.values())
        exp_model.add_cpds(*pit_cpds_dict.values())

        return exp_model

    def get_evidence_dict(self):
        """
        generated the evidence dict
        """
        evidence_dict = {}
        for pit_location in self.pit_locations:
            evidence_dict[f'P{pit_location.x}{pit_location.y}'] = True
        for no_pit_location in self.no_pit_locations:
            evidence_dict[f'P{no_pit_location.x}{no_pit_location.y}'] = False
        for breeze_location in self.breeze_locations:
            evidence_dict[f'B{breeze_location.x}{breeze_location.y}'] = True
        for no_breeze_location in self.no_breeze_locations:
            evidence_dict[f'B{no_breeze_location.x}{no_breeze_location.y}'] = False
        
        return evidence_dict

    def check_pit_probability_of(self, *, location, bayesian_model, evidence_dict):
        """
        returns the probability of being a pit for the input location
        """
        exp_infer = VariableElimination(bayesian_model)

        return exp_infer.query([f'P{location.x}{location.y}'], evidence=evidence_dict, show_progress=False).values[0]

    def check_safe_probability_of(self, *, location, bayesian_model, evidence_dict):
        """
        returns the probability of being safe for the input location
        """
        wumpus_probability = 0.0

        if location in self.possible_wumpus_locations:
            wumpus_probability = 1 / len(self.possible_wumpus_locations)

        pit_probability = self.check_pit_probability_of(location=location, bayesian_model=bayesian_model, evidence_dict=evidence_dict)
    
        return (1 - pit_probability) * (1 - wumpus_probability)

class HuntWumpusHybridAgent(object):
    """
    This class defines an agent for the wumpus world that does logical inference.
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self.plan_actions = list()
        self.previous_agent_action = None

    def get_next_action_from(self, *, percept):
        """
        takes in input the current percpept from the agent and returns the best action to perform 
        """
        self.kb.update_with(previous_agent_action=self.previous_agent_action, current_percept=percept)

        if percept.glitter:
            if self.plan_actions:
                self.plan_actions = []

            self.plan_actions.append(Hunter.Actions.GRAB)
            # here we plan a route that can have more possible wumpus locations, which i can shoot if the arrow is available. 
            # the possible wumpus locations are added as allowed cells and also as possible wumpus. 
            # A* will take care if shooting the wumpus (even if it is only a possible wumpus) is efficient.
            self.plan_actions.extend(self.plan_route(to_goal_locations=self.kb.exit_locations, 
                                                     with_allowed_locations=self.kb.get_safe_from_pit_but_possible_wumpus_locations(), 
                                                     from_agent_location=self.kb.agent_location, 
                                                     from_agent_orientation=self.kb.agent_orientation, 
                                                     is_arrow_available=self.kb.is_arrow_available, 
                                                     with_possible_wumpus_locations=self.kb.possible_wumpus_locations))
            self.plan_actions.append(Hunter.Actions.CLIMB)
            
        if not self.plan_actions:
            
            # here we plan a route to a safe unvisited cell. 
            # If we know already for sure where the wumpus is and he is not in a cell with a pit we add his position 
            # to the safe cell and wumpus_location and A* will take care if shooting the wumpus is more efficient 
            # then looking for another safe unvisited cell
            if self.kb.known_wumpus_location is not None and self.kb.known_wumpus_location in self.kb.no_pit_locations:
                goal_locations = self.kb.get_safe_unvisited_locations()
                goal_locations.add(self.kb.known_wumpus_location)
                self.plan_actions.extend(self.plan_route(to_goal_locations=goal_locations, 
                                                         with_allowed_locations=self.kb.get_safe_from_pit_but_possible_wumpus_locations(), 
                                                         from_agent_location=self.kb.agent_location, 
                                                         from_agent_orientation=self.kb.agent_orientation, 
                                                         is_arrow_available=self.kb.is_arrow_available, 
                                                         with_possible_wumpus_locations=self.kb.possible_wumpus_locations))
            else:
                self.plan_actions.extend(self.plan_route(to_goal_locations=self.kb.get_safe_unvisited_locations(), 
                                                         with_allowed_locations=self.kb.get_safe_locations(), 
                                                         from_agent_location=self.kb.agent_location, 
                                                         from_agent_orientation=self.kb.agent_orientation, 
                                                         is_arrow_available=self.kb.is_arrow_available, 
                                                         with_possible_wumpus_locations=self.kb.possible_wumpus_locations))
            
        if not self.plan_actions and self.kb.is_arrow_available and len(self.kb.possible_wumpus_locations) > 1:
            
            possible_wumpus_locations = self.kb.possible_wumpus_locations.intersection(self.kb.no_pit_locations)

            if possible_wumpus_locations:
            
                self.plan_actions.extend(self.plan_route(to_goal_locations=possible_wumpus_locations, 
                                                         with_allowed_locations=self.kb.get_safe_locations().union(possible_wumpus_locations), 
                                                         from_agent_location=self.kb.agent_location, 
                                                         from_agent_orientation=self.kb.agent_orientation, 
                                                         is_arrow_available=self.kb.is_arrow_available, 
                                                         with_possible_wumpus_locations=possible_wumpus_locations))

        if not self.plan_actions:

            self.kb.update_pit_locations()

            # add bayes network
            max_safe_probability = 0.0
            max_safe_locations = set()

            bayesian_model = self.kb.get_bayesian_model()
            evidence_dict = self.kb.get_evidence_dict()

            if self.kb.is_arrow_available and self.kb.possible_wumpus_locations:
                for fringe_location in self.kb.fringe_locations.difference(self.kb.pit_locations).difference(self.kb.no_pit_locations):
                    if (safe_probability := 1 - self.kb.check_pit_probability_of(location=fringe_location, 
                                                                                bayesian_model=bayesian_model, 
                                                                                evidence_dict=evidence_dict)) > max_safe_probability:
                        max_safe_probability = safe_probability
                        max_safe_locations = set([fringe_location])

                    elif safe_probability == max_safe_probability:
                        max_safe_locations.add(fringe_location) 

            else:
                for fringe_location in self.kb.fringe_locations.difference(self.kb.pit_locations).difference(self.kb.no_pit_locations):
                    if (safe_probability := self.kb.check_safe_probability_of(location=fringe_location, 
                                                                            bayesian_model=bayesian_model, 
                                                                            evidence_dict=evidence_dict)) > max_safe_probability:
                        max_safe_probability = safe_probability
                        max_safe_locations = set([fringe_location])

                    elif safe_probability == max_safe_probability:
                        max_safe_locations.add(fringe_location)

            if max_safe_probability > MIN_SAFE_PROBABILITY_TRESHOLD:
                self.plan_actions.extend(self.plan_route(to_goal_locations=max_safe_locations, 
                                                        with_allowed_locations=self.kb.get_safe_from_pit_but_possible_wumpus_locations().union(max_safe_locations),
                                                        from_agent_location=self.kb.agent_location,
                                                        from_agent_orientation=self.kb.agent_orientation,
                                                        is_arrow_available=self.kb.is_arrow_available,
                                                        with_possible_wumpus_locations=self.kb.possible_wumpus_locations))

        if not self.plan_actions:
            self.plan_actions.extend(self.plan_route(to_goal_locations=self.kb.exit_locations,
                                                     with_allowed_locations=self.kb.get_safe_from_pit_but_possible_wumpus_locations(),
                                                     from_agent_location=self.kb.agent_location,
                                                     from_agent_orientation=self.kb.agent_orientation,
                                                     is_arrow_available=self.kb.is_arrow_available,
                                                     with_possible_wumpus_locations=self.kb.possible_wumpus_locations))
            self.plan_actions.append(Hunter.Actions.CLIMB)
            
        action = self.plan_actions[0]
        del self.plan_actions[0]
        self.previous_agent_action = action
        return action

    def plan_route(self, *, to_goal_locations, with_allowed_locations, from_agent_location, from_agent_orientation, 
                            is_arrow_available, with_possible_wumpus_locations):
        """
        calculate the best plan of actions to reach the closest goal location from the agent location
        """
        goal_locations = to_goal_locations
        allowed_locations = with_allowed_locations
        agent_location = from_agent_location
        agent_orientation = from_agent_orientation
        possible_wumpus_locations = with_possible_wumpus_locations

        if not goal_locations:
            return []

        def goal_test(state):
            return state.agent_location in state.goal_locations

        useful_actions = list(filter(lambda x: x not in {Hunter.Actions.GRAB, Hunter.Actions.CLIMB}, Hunter.Actions))

        def heuristic_func(state):
            return min([smart_manhattan_distance(state.agent_location, destination=goal_location, safe_locations=state.safe_locations) for goal_location in state.goal_locations])
        
        problem = HuntWumpusProblem(safe_locations=allowed_locations,
                                    goal_locations=goal_locations,
                                    possible_actions=useful_actions,
                                    is_goal_state=goal_test,
                                    heuristic_func=heuristic_func,
                                    agent_location=agent_location,
                                    agent_orientation=agent_orientation,
                                    is_arrow_available=is_arrow_available,
                                    wumpus_locations=possible_wumpus_locations) 
        return self.astar_search(problem=problem)

    def astar_search(self, *, problem):
        """
        performs an offline A* search for a problem of type HuntWumpusProblem
        """

        if (problem.is_goal_state(problem.initial_state)):
            return []

        frontier = PriorityQueue()
        reached = {} # {state: int}
        solution = HuntWumpusNode(problem.initial_state, math.inf)

        initial_node = HuntWumpusNode(problem.initial_state)
        frontier.put(initial_node)
        reached[initial_node.state] = initial_node.get_cost_heuristic_sum()

        while not frontier.empty() and (node := frontier.get()).get_cost_heuristic_sum() < solution.get_cost_heuristic_sum():
            # this modification is needed since the PriorityQueue that we use doesn’t update the value  of a node that is 
            # already present when put(node) is executed. It will just add the cheaper one in a lower position. 
            # When backtracking occurs it is not needed to expand a node that was already expanded with a lower 
            # value, therefore we can safely skip it.
            if node.get_cost_heuristic_sum() > reached[node.state]:
                continue

            childs = [problem.get_child_from(node, with_action= action) for action in problem.get_best_actions_for(node.state)]

            for child in childs:
                if (child.state not in reached) or (child.get_cost_heuristic_sum() < reached[child.state]):
                    reached[child.state] = child.get_cost_heuristic_sum()
                    frontier.put(child)

                    if problem.is_goal_state(child.state) and child.get_cost_heuristic_sum() < solution.get_cost_heuristic_sum():
                        solution = child

        sequence_actions = []
        sequence_actions = problem.unwrap_solution(solution)
        if sequence_actions:
            return sequence_actions  
        else:
            return []