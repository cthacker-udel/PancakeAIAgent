from __future__ import annotations
from typing import List

class Pancake:
    def __init__(self: Pancake, size: int, burnt: bool) -> None:
        self.size = size
        self.burnt = burnt


class PancakeState:
    def __init__(self: PancakeState, pancakes: List[Pancake]):
        self.pancakes = pancakes
        self.num_pancakes = len(pancakes)


class StateGraph:
    def __init__(self: StateGraph, state: PancakeState):
        self.root_state: PancakeState = state
        self.moves: List[StateGraph] =[]
        for i in self.root_state.num_pancakes:
            


class PancakeFlippingSolver:
    def __init__(self: PancakeFlippingSolver, algorithm: str, pancake_string: str) -> None:
        parsed_pancakes: List[Pancake] = []
        for i in range(0, len(pancake_string), 2):
            parsed_pancakes.append(Pancake(int(pancake_string[i]), pancake_string[i + 1].lower() == 'b'))
        


def main():
    print(' in main')
    pancake_order = input()
    [pancakes, algo] = pancake_order.split('-')
    solver = PancakeFlippingSolver(algo, pancakes)




if __name__ =='__main__':
    main()