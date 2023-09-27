"""
TODO: Finish These

- [ ] Create sound structure for state graph
- [ ] Implement BFS Algorithm
- [ ] Run tests on BFS Algorithm
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
from uuid import uuid4
from unittest import TestCase

test_cases = [
    ["1w2b3b4w-b", "1w2b|3b4w\n2w|1b3b4w\n2b1b|3b4w\n1w2w3b|4w\n3w|2b1b4w\n3b2b1b|4w\n1w2w3w4w"],
    ["1b2b3w4b-a", "1b2b|3w4b g:0, h:0\n2w|1w3w4b g:2, h:2\n2b1w3w4b| g:3, h:2\n4w|3b1b2w g:7, h:4\n4b3b1b2w| g:8, h:4\n2b1w|3w4w g:12, h:2\n1b|2w3w4w g:14, h:0\n1w2w3w4w g:15, h:0"],
    ["1w2b3w4b-a", "1w2b|3w4b g:0, h:0\n2w|1b3w4b g:2, h:2\n2b1b3w4b| g:3, h:2\n4w|3b1w2w g:7, h:4\n4b3b1w2w| g:8, h:4\n4b3b1w2w| g:8, h:4\n1w2w3w4w g:14, h:0"],
    ["1w2b3w4b-b", "1w2b|3w4b\n2w|1b3w4b\n2b1b|3w4b\n1w2w3w4b|\n4w|3b2b1b\n4b3b2b1b|\n1w2w3w4w"],
    ["1b2b3w4b-a", "1b2b|3w4b g:0, h:0\n2w|1w3w4b g:2, h:2\n2b1w3w4b| g:3, h:2\n4w|3b1b2w g:7, h:4\n4b3b1b2w| g:8, h:4\n2b1w|3w4w g:12, h:2\n1b|2w3w4w g:14, h:0\n1w2w3w4w g:15, h:0"],
    ["1b2w3b4b-a", "1b2w3b4b| g:0, h:0\n4w|3w2b1w g:4, h:4\n4b3w|2b1w g:5, h:4\n3b|4w2b1w g:7, h:4\n3w4w|2b1w g:8, h:4\n4b3b2b1w| g:10, h:4\n1b|2w3w4w g:14, h:0\n1w2w3w4w g:15, h:0"]
]


class Algorithm(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """
    BFS = 0,
    DFS = 1,
    A_STAR = 2


available_algorithms: dict[str, Algorithm] = {
    'b': Algorithm.BFS,
    'd': Algorithm.DFS,
    'a': Algorithm.A_STAR
}


class Pancake:
    def __init__(self: Pancake, size: int, burnt: bool) -> None:
        self.size = size
        self.burnt = burnt

    def flip(self: Pancake) -> None:
        self.burnt = not self.burnt

    def clone(self: Pancake) -> Pancake:
        return Pancake(self.size, self.burnt)

    def __str__(self: Pancake) -> str:
        return f'{self.size}{self.burnt}'


class PancakeState:
    def __init__(self: PancakeState, pancakes: List[Pancake]):
        self.pancakes = pancakes
        self.num_pancakes = len(pancakes)
        self.state_id = str(uuid4())
        self.explored = False
        self.parent: Optional[PancakeState] = None
        self.flipped = 0

    def clone(self: PancakeState) -> PancakeState:
        cloned = PancakeState([x.clone() for x in self.pancakes])
        cloned.num_pancakes = len(self.pancakes)
        cloned.state_id = str(uuid4())
        cloned.explored = False
        return cloned

    def explore(self: PancakeState) -> None:
        self.explored = True

    def flip(self: PancakeState, flip_index: int) -> PancakeState:
        cloned_state = self.clone()
        cloned_state.flipped = flip_index
        flipped_part = cloned_state.pancakes[0:flip_index][::-1]
        base_part = cloned_state.pancakes[flip_index:]
        cloned_state.pancakes = flipped_part + base_part
        for i in range(0, flip_index):
            cloned_state.pancakes[i].flip()

        return cloned_state

    def generate_possible_moves(self: PancakeState) -> List[PancakeState]:
        potential_moves: List[PancakeState] = []
        for i in range(1, len(self.pancakes)):
            potential_moves.append(self.flip(i))

        return potential_moves

    def __str__(self: PancakeState) -> str:
        if self.flipped == 0:
            return ''.join([f'{x.size}{"b" if x.burnt else "w"}' for x in self.pancakes])
        else:
            steps = [
                f'{x.size}{"b" if x.burnt else "w"}' for x in self.pancakes]
            steps.insert(self.flipped, '|')
            return ''.join(steps)


class StateGraph:
    def __init__(self: StateGraph, state: PancakeState, algorithm: Algorithm) -> None:
        self.root_state: PancakeState = state
        self.moves: List[StateGraph] = []
        self.algorithm = algorithm

    def is_goal(self: StateGraph, state: PancakeState) -> bool:
        state_pancakes = state.pancakes
        are_pancakes_descending = sorted([x.size for x in state_pancakes]) == [
            x.size for x in state_pancakes]
        are_pancakes_burnt_sides_down = all(
            [not x.burnt for x in state_pancakes])
        return are_pancakes_burnt_sides_down and are_pancakes_descending

    def search_for_goal(self: StateGraph) -> PancakeState | None:
        if self.algorithm == Algorithm.BFS:
            return self.bfs_algorithm()
        elif self.algorithm == Algorithm.DFS:
            return None
        else:
            # A*
            return None

    def bfs_algorithm(self: StateGraph) -> PancakeState:
        bfs_queue: List[PancakeState] = []
        self.root_state.explore()
        explored_states: Set[str] = set([str(self.root_state)])
        bfs_queue.append(self.root_state)
        while len(bfs_queue) > 0:
            parent_state = bfs_queue.pop()
            if self.is_goal(parent_state):
                return parent_state
            else:
                expansion_nodes = parent_state.generate_possible_moves()
                for each_node in expansion_nodes:
                    if not each_node.explored and not str(each_node) in explored_states:
                        each_node.explore()
                        explored_states.add(str(each_node))
                        each_node.parent = parent_state
                        bfs_queue.append(each_node)
                    else:
                        each_node.explore()
        # todo
        return self.root_state


class PancakeFlippingSolver:
    """_summary_
    """

    def __init__(self: PancakeFlippingSolver, algorithm: str, pancake_string: str) -> None:
        parsed_pancakes: List[Pancake] = []
        for i in range(0, len(pancake_string), 2):
            parsed_pancakes.append(
                Pancake(int(pancake_string[i]), pancake_string[i + 1].lower() == 'b'))
        if algorithm not in available_algorithms:
            raise ValueError(
                f"Invalid algorithm specified, only options are {','.join([x for x in available_algorithms.keys()])}")
        self.algorithm = available_algorithms[algorithm]
        pancake_graph = StateGraph(PancakeState(
            parsed_pancakes), available_algorithms[algorithm])
        steps = pancake_graph.search_for_goal()

        if steps is not None:
            compiled_steps = self.compile_steps(steps)
            print(compiled_steps)

    def compile_steps(self: PancakeFlippingSolver, state: PancakeState | None) -> List[str]:
        steps = []
        while state is not None:
            steps.append(str(state))
            state = state.parent
        return steps


def main(run_input: bool = False):
    """_summary_
    """
    print(' in main')
    if run_input:
        pancake_order = input()
        [pancakes, algo] = pancake_order.split('-')
        solver = PancakeFlippingSolver(algo, pancakes)
    else:
        tester = TestCase()
        for each_test_case in test_cases:
            [pancakes, algo] = each_test_case[0].split('-')
            solver = PancakeFlippingSolver(algo, pancakes)
            tester.assertEqual(each_test_case[0], each_test_case[1])
        print('Passed all tests!')


if __name__ == '__main__':
    main()
