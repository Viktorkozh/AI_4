#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Представьте, что вы разрабатываете систему
# для автоматического управления инвестициями, где дерево решений используется
# для представления последовательности инвестиционных решений
# и их потенциальных исходов. Цель состоит в том, чтобы найти
# наилучший исход (максимальную прибыль) на определённой глубине принятия решений,
# учитывая ограниченные ресурсы и время на анализ.

import math
from collections import deque
from typing import Any, Iterator, List, Optional, Union


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


def depth_limited_search(problem: Problem, limit: int) -> Union[int, None]:
    frontier = LIFOQueue([Node(problem.initial)])
    max_value = -123
    found_value = False

    while frontier:
        node = frontier.pop()

        if len(node) == limit:
            max_value = max(max_value, node.state.value)
            found_value = True

        if len(node) < limit:
            for child in Node.expand(problem, node):
                frontier.append(child)

    return max_value if found_value else None


class BinaryTreeNode:
    def __init__(
        self,
        value: int,
        left: Optional["BinaryTreeNode"] = None,
        right: Optional["BinaryTreeNode"] = None,
    ):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"<{self.value}>"


class InvestProblem(Problem):
    def __init__(self, initial: BinaryTreeNode) -> None:
        super().__init__(initial, None)

    def actions(self, state: BinaryTreeNode) -> List[BinaryTreeNode]:
        actions = []
        if state.left:
            actions.append(state.left)
        if state.right:
            actions.append(state.right)
        return actions

    def result(self, state: BinaryTreeNode, action: BinaryTreeNode) -> BinaryTreeNode:
        return action


def main() -> None:
    root = BinaryTreeNode(
        3,
        BinaryTreeNode(1, BinaryTreeNode(0), None),
        BinaryTreeNode(5, BinaryTreeNode(4), BinaryTreeNode(6)),
    )
    limit = 2

    problem = InvestProblem(root)

    max_value = depth_limited_search(problem, limit)

    if max_value is not None:
        print(f"Максимальное значение на указанной глубине: {max_value}")
    else:
        print("Нет значений на указанной глубине.")


if __name__ == "__main__":
    main()
