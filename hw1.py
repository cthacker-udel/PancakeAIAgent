"""
TODO: Finish These

- [X] Create sound structure for state graph
- [X] Implement BFS Algorithm
- [X] Run tests on BFS Algorithm
- [ ] Implement DFS Algorithm
- [ ] Run tests on DFS Algorithm
- [ ] Add in heuristic capabilities into nodes
- [ ] Run tests on both suites to ensure program is stable
- [ ] Implement A* algorithm
- [ ] Run full test suite to determine if output is correct

"""

from __future__ import annotations
from typing import List, Optional, Set
from enum import Enum
from unittest import TestCase

#############################
# TEST CASES
#############################
test_cases = [
    ["1w2b3b4w-b", "1w2b|3b4w\n2w|1b3b4w\n2b1b|3b4w\n1w2w3b|4w\n3w|2b1b4w\n3b2b1b|4w\n1w2w3w4w"],
    ["1w2b3w4b-b", "1w2b|3w4b\n2w|1b3w4b\n2b1b|3w4b\n1w2w3w4b|\n4w|3b2b1b\n4b3b2b1b|\n1w2w3w4w"],
    ["1b2b3w4b-a", "1b2b|3w4b g:0, h:0\n2w|1w3w4b g:2, h:2\n2b1w3w4b| g:3, h:2\n4w|3b1b2w g:7, h:4\n4b3b1b2w| g:8, h:4\n2b1w|3w4w g:12, h:2\n1b|2w3w4w g:14, h:0\n1w2w3w4w g:15, h:0"],
    ["1w2b3w4b-a", "1w2b|3w4b g:0, h:0\n2w|1b3w4b g:2, h:2\n2b1b3w4b| g:3, h:2\n4w|3b1w2w g:7, h:4\n4b3b1w2w| g:8, h:4\n4b3b1w2w| g:8, h:4\n1w2w3w4w g:14, h:0"],
    ["1b2b3w4b-a", "1b2b|3w4b g:0, h:0\n2w|1w3w4b g:2, h:2\n2b1w3w4b| g:3, h:2\n4w|3b1b2w g:7, h:4\n4b3b1b2w| g:8, h:4\n2b1w|3w4w g:12, h:2\n1b|2w3w4w g:14, h:0\n1w2w3w4w g:15, h:0"],
    ["1b2w3b4b-a", "1b2w3b4b| g:0, h:0\n4w|3w2b1w g:4, h:4\n4b3w|2b1w g:5, h:4\n3b|4w2b1w g:7, h:4\n3w4w|2b1w g:8, h:4\n4b3b2b1w| g:10, h:4\n1b|2w3w4w g:14, h:0\n1w2w3w4w g:15, h:0"]
]


class Algorithm(Enum):
    """
    Represents all the available algorithms this solver can implement

    Args:
        Enum (_type_): Used for constructing an enum type
    """
    BFS = 0,
    DFS = 1,
    A_STAR = 2


################################
# ALL AVAILABLE ALGORITHMS
################################
available_algorithms: dict[str, Algorithm] = {
    'b': Algorithm.BFS,
    'd': Algorithm.DFS,
    'a': Algorithm.A_STAR
}


class Pancake:
    """
    Represents a single pancake in the pancake problem
    """

    def __init__(self: Pancake, size: int, burnt: bool) -> None:
        """
        Initializes the pancake, setting it's size, and whether it was burned

        Args:
            self (Pancake): The internal pancake instance
            size (int): The size (order) of the pancake
            burnt (bool): Whether the pancake is burnt (as we want all white sides up)
        """
        self.size = size
        self.burnt = burnt

    def flip(self: Pancake) -> None:
        """
        Flips the pancake, setting it's `burnt` member to the opposite of what it is (burnt --> white, white --> burnt)

        Args:
            self (Pancake): The internal pancake instance
        """
        self.burnt = not self.burnt

    def clone(self: Pancake) -> Pancake:
        """
        Clones the pancake class, avoiding accidental mutation

        Args:
            self (Pancake): The internal pancake instance

        Returns:
            Pancake: The cloned pancake instance
        """
        return Pancake(self.size, self.burnt)

    def __str__(self: Pancake) -> str:
        """
        A stringified version of the Pancake

        Args:
            self (Pancake): The pancake instance

        Returns:
            str: The stringified version of the pancake
        """
        return f'{self.size}{self.burnt}'


class PancakeState:
    """
    Represents a node in the state graph
    """

    def __init__(self: PancakeState, pancakes: List[Pancake]):
        """
        Initializes the PancakeState, setting it's fields such as the array of pancakes in the state itself (the node within a state, which contains all the pancakes),
        and whether the node has been explored ("expanded")

        Args:
            self (PancakeState): The internal state, used for mutation
            pancakes (List[Pancake]): The list of pancakes to initialize the pancakes array as
        """
        self.pancakes = pancakes
        self.num_pancakes = len(pancakes)
        self.explored = False
        self.parent: Optional[PancakeState] = None
        self.flipped = 0

    def clone(self: PancakeState) -> PancakeState:
        """
        Deep clones the state, allowing for no accidental mutation to take place

        Args:
            self (PancakeState): The internal state

        Returns:
            PancakeState: The cloned state, a new instance of the PancakeState class
        """
        cloned = PancakeState([x.clone() for x in self.pancakes])
        cloned.num_pancakes = len(self.pancakes)
        cloned.explored = False
        cloned.flipped = self.flipped
        return cloned

    def explore(self: PancakeState) -> None:
        """
        Marks the node as explored

        Args:
            self (PancakeState): The internal states
        """
        self.explored = True

    def flip(self: PancakeState, flip_index: int) -> PancakeState:
        """
        Executes when the user decides to flip the pancakes, effectively clones all the classes, mutates their members, and returns
        the cloned state

        Args:
            self (PancakeState): The internal state, which is cloned and mutated
            flip_index (int): The index where the "flip" is taking place

        Returns:
            PancakeState: The mutated & cloned state
        """
        cloned_state = self.clone()
        cloned_state.flipped = flip_index
        flipped_part = cloned_state.pancakes[0:flip_index][::-1]
        base_part = cloned_state.pancakes[flip_index:]
        cloned_state.pancakes = flipped_part + base_part
        for i in range(0, flip_index):
            cloned_state.pancakes[i].flip()

        return cloned_state

    def generate_possible_moves(self: PancakeState) -> List[PancakeState]:
        """
        Generates all possible moves that could be made, aka all the possible flips the user can make on the state

        Args:
            self (PancakeState): The internal state

        Returns:
            List[PancakeState]: The list of potential moves the user can make
        """
        potential_moves: List[PancakeState] = []
        for i in range(1, len(self.pancakes) + 1):
            potential_moves.append(self.flip(i))

        return potential_moves

    def __str__(self: PancakeState) -> str:
        """
        Converts the pancake state into a stringified version of it, which is used in the submission, and generally for a better representation
        then a memory address

        Args:
            self (PancakeState): The internal state, used for referencing values while stringifying the state

        Returns:
            str: The stringified version of the state
        """
        if self.flipped == 0:
            return ''.join([f'{x.size}{"b" if x.burnt else "w"}' for x in self.pancakes])
        else:
            steps = [
                f'{x.size}{"b" if x.burnt else "w"}' for x in self.pancakes]
            steps.insert(self.flipped, '|')
            return ''.join(steps)


class StateGraph:
    """
    Represents the graph of the state of the pancake problem
    """

    def __init__(self: StateGraph, state: PancakeState, algorithm: Algorithm) -> None:
        """
        Initializes the state graph with the supplied state, and the supplied algorithm

        Args:
            self (StateGraph): The internal state, used for setting members
            state (PancakeState): The state to initialize as the root node in this graph
            algorithm (Algorithm): The algorithm to execute with this graph
        """
        self.root_state: PancakeState = state
        self.moves: List[StateGraph] = []
        self.algorithm = algorithm

    def is_goal(self: StateGraph, state: PancakeState) -> bool:
        """
        Whether the state corresponds to the goal state

        Args:
            self (StateGraph): The internal graph state
            state (PancakeState): The last step in the state graph

        Returns:
            bool: Whether the state supplied as an argument (`state`) is the goal state
        """
        state_pancakes = state.pancakes
        are_pancakes_descending = sorted([x.size for x in state_pancakes]) == [
            x.size for x in state_pancakes]
        are_pancakes_burnt_sides_down = all(
            [not x.burnt for x in state_pancakes])
        return are_pancakes_burnt_sides_down and are_pancakes_descending

    def search_for_goal(self: StateGraph) -> PancakeState | None:
        """
        Basically manages which algorithm to execute, if the algorithm supplied is 'b', then execute BFS, and so on.

        Args:
            self (StateGraph): The internal state

        Returns:
            PancakeState | None: The state node corresponding to the last step in the solution
        """
        if self.algorithm == Algorithm.BFS:
            return self.bfs_algorithm()
        elif self.algorithm == Algorithm.DFS:
            return None
        else:
            # A*
            return None

    def bfs_algorithm(self: StateGraph) -> PancakeState:
        """
        Runs the BFS algorithm on the internal state graph that was initialized in the constructor

        Args:
            self (StateGraph): The internal state graph instance

        Returns:
            PancakeState: The state node that corresponds to the solved problem
        """
        bfs_queue: List[PancakeState] = []
        self.root_state.explore()
        explored_states: Set[str] = set([str(self.root_state)])
        bfs_queue.append(self.root_state)
        while len(bfs_queue) > 0:
            parent_state = bfs_queue.pop(0)
            if self.is_goal(parent_state):
                return parent_state

            expansion_nodes = parent_state.generate_possible_moves()
            for each_node in expansion_nodes:
                if not each_node.explored and not str(each_node) in explored_states:
                    each_node.explore()
                    explored_states.add(str(each_node))
                    each_node.parent = parent_state
                    bfs_queue.append(each_node)
                else:
                    each_node.explore()
        return self.root_state


class PancakeFlippingSolver:
    """
    Solver for all algorithms relating to the pancake flipping problem
    """

    def __init__(self: PancakeFlippingSolver, algorithm: str, pancake_string: str) -> None:
        ########################
        # PARSING INPUT
        ########################
        parsed_pancakes: List[Pancake] = []
        for i in range(0, len(pancake_string), 2):
            parsed_pancakes.append(
                Pancake(int(pancake_string[i]), pancake_string[i + 1].lower() == 'b'))
        if algorithm not in available_algorithms:
            raise ValueError(
                f"Invalid algorithm specified, only options are {','.join(list(x for x in available_algorithms))}")
        self.algorithm: Algorithm = available_algorithms[algorithm]

        #########################
        # SETTING STATE GRAPH
        #########################
        self.state_graph = StateGraph(PancakeState(
            parsed_pancakes), available_algorithms[algorithm])
        self.stringified_steps = ''

    def run_algorithm(self: PancakeFlippingSolver) -> PancakeState | None:
        """
            Runs the specified algorithm supplied in the constructor

        Args:
            self (PancakeFlippingSolver): The internal instance
        """
        steps = self.state_graph.search_for_goal()
        if steps is not None:
            self.stringified_steps = self.stringify_steps(steps)
            return steps
        return None

    def stringify_steps(self: PancakeFlippingSolver, state: PancakeState | None) -> str:
        """
        Stringifies all the steps made in the algorithmic process

        Args:
            self (PancakeFlippingSolver): The internal state
            state (PancakeState | None): The state we are stringifying it's steps, by traversing up the parents (the steps it made before it)

        Returns:
            List[str]: The steps separated by a \\n character
        """
        if state is None:
            return ''

        steps: List[PancakeState] = []
        while state is not None:
            steps.append(state.clone())
            state = state.parent

        cloned_steps = [x.clone() for x in steps]
        for ind, each_cloned_step in enumerate(cloned_steps):
            if '|' in str(each_cloned_step):
                cloned_steps[ind] = each_cloned_step.flip(
                    each_cloned_step.flipped)

        stringified_cloned_steps = [
            str(x) for x in cloned_steps[:-1][::-1]] + [str(cloned_steps[0].flip(cloned_steps[0].flipped)).replace('|', '')]
        return '\n'.join(stringified_cloned_steps)


def main(run_input: bool = False):
    """
    The main loop, allows for unit testing and taking in raw input and processing its

    Args:
        run_input (bool, optional): Whether to run the input loop or the test cases. Defaults to False.
    """
    if run_input:
        pancake_order = input()
        [pancakes, algo] = pancake_order.split('-')
        solver = PancakeFlippingSolver(algo, pancakes)
    else:
        tester = TestCase()
        for each_test_case in test_cases:
            print(f'\n--------- TESTING {each_test_case[0]} ---------\n')
            [pancakes, algo] = each_test_case[0].split('-')
            solver = PancakeFlippingSolver(algo, pancakes)
            last_step = solver.run_algorithm()
            steps = solver.stringify_steps(last_step)
            tester.assertEqual(steps, each_test_case[1])
            print(
                f'--------- {each_test_case[0]} PASSED ---------\n')


if __name__ == '__main__':
    main()
