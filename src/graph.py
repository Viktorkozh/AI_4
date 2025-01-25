#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import json
from typing import Dict, Set, List, Optional, Tuple, Any, Iterator, Union
from collections import deque


class Problem:
    initial: Any
    goal: Optional[Any]

    def __init__(
        self, initial: Any = None, goal: Optional[Any] = None, **kwds: Any
    ) -> None:
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state: Any) -> List[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def is_goal(self, state: Any) -> Any:
        return state == self.goal

    def action_cost(self, s: Any, a: Any, s1: Any) -> float:
        return 1

    def h(self, node: "Node") -> float:
        return 0

    def __str__(self) -> str:
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    state: Any
    parent: Optional["Node"]
    action: Optional[Any]
    path_cost: float

    def __init__(
        self,
        state: Any,
        parent: Optional["Node"] = None,
        action: Optional[Any] = None,
        path_cost: float = 0,
    ) -> None:
        self.__dict__.update(
            state=state, parent=parent, action=action, path_cost=path_cost
        )

    def __repr__(self) -> str:
        return "<{}>".format(self.state)

    def __len__(self) -> int:
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other: "Node") -> bool:
        return self.path_cost < other.path_cost

    @staticmethod
    def is_cycle(node: "Node") -> bool:
        parent = node.parent
        while parent:
            if parent.state == node.state:
                return True
            parent = parent.parent
        return False

    failure: "Node"
    cutoff: "Node"

    @staticmethod
    def expand(problem: Problem, node: "Node") -> Iterator["Node"]:
        s = node.state
        for action in problem.actions(s):
            s1 = problem.result(s, action)
            cost = node.path_cost + problem.action_cost(s, action, s1)
            yield Node(s1, node, action, cost)


Node.failure = Node("failure", path_cost=math.inf)
Node.cutoff = Node("cutoff", path_cost=math.inf)


LIFOQueue = deque


def depth_limited_search(problem: Problem, limit: int = 10) -> Union[Node, "Node"]:
    frontier = LIFOQueue([Node(problem.initial)])
    result = Node.failure

    while frontier:
        node = frontier.pop()

        if problem.is_goal(node.state):
            return node
        elif len(node) >= limit:
            result = Node.cutoff
        elif not Node.is_cycle(node):
            for child in Node.expand(problem, node):
                frontier.append(child)

    return result


class CityProblem(Problem):
    cities: Dict[str, str]
    distances: Dict[Tuple[str, str], float]
    initial: str
    goal: str

    def __init__(
        self,
        cities: Dict[str, str],
        distances: Dict[Tuple[str, str], float],
        initial: str,
        goal: str,
    ) -> None:
        super().__init__(initial=initial, goal=goal)
        self.cities = cities
        self.distances = distances

    def actions(self, state: str) -> List[str]:
        return [target for target in self.cities if (state, target) in self.distances]

    def result(self, state: str, action: str) -> str:
        return action

    def action_cost(self, s: str, a: str, s1: str) -> float:
        return self.distances.get((s, s1), 1.0)


def reconstruct_path(came_from: Dict[str, Optional[str]], current: str) -> List[str]:
    path: List[str] = []
    while current is not None:
        path.append(current)
        next_current = came_from.get(current)
        if next_current is None:
            break
        current = next_current
    path.reverse()
    return path


def main() -> None:
    with open("elem.json", "r", encoding="utf-8") as file:
        data: List[Dict[str, Any]] = json.load(file)

    selected_ids: Set[str] = {"8", "9", "2", "15", "6", "1", "3", "7", "13", "18"}

    cities: Dict[str, str] = {}
    distances: Dict[Tuple[str, str], float] = {}

    for item in data:
        if "label" in item["data"]:
            if item["data"]["id"] in selected_ids:
                cities[item["data"]["id"]] = item["data"]["label"]
        elif "source" in item["data"]:
            source: str = item["data"]["source"]
            target: str = item["data"]["target"]
            if source in selected_ids and target in selected_ids:
                weight: float = float(item["data"]["weight"])
                distances[(source, target)] = weight
                distances[(target, source)] = weight

    start_city: str = "8"
    goal_city: str = "15"

    problem = CityProblem(cities, distances, start_city, goal_city)
    solution: Optional[Node] = depth_limited_search(problem)

    if solution is None:
        print("Решение не найдено.")
    else:
        path: List[str] = []
        current: Optional[Node] = solution
        while current is not None:
            path.append(current.state)
            current = current.parent
        path.reverse()
        print("Кратчайший путь:")
        print(" -> ".join(cities[city] for city in path))


if __name__ == "__main__":
    main()
