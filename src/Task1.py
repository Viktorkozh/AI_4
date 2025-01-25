#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Вы работаете над разработкой системы навигации для робота-пылесоса.
# Робот способен передвигаться по различным комнатам в доме,
# но из-за ограниченности ресурсов (например, заряда батареи) и времени на уборку,
# важно эффективно выбирать путь. Ваша задача - реализовать алгоритм,
# который поможет роботу определить, существует ли путь к целевой комнате,
# не превышая заданное ограничение по глубине поиска.
# Дано дерево, где каждый узел представляет собой комнату в доме.
# Узлы связаны в соответствии с возможностью перемещения робота
# из одной комнаты в другую. Необходимо определить, существует ли путь
# от начальной комнаты (корень дерева) к целевой комнате (узел с заданным значением),
# так, чтобы робот не превысил лимит по глубине перемещения.

import math
from collections import deque
from typing import Any, Iterator, List, Optional, TypeVar, Union

T = TypeVar("T")


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


LIFOQueue = deque


Node.failure = Node("failure", path_cost=math.inf)
Node.cutoff = Node("cutoff", path_cost=math.inf)


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


class RoombaNavProblem(Problem):
    rooms_tree: Any
    target_room: Any
    depth_limit: int

    def __init__(self, rooms_tree: Any, target_room: Any, depth_limit: int) -> None:
        super().__init__(
            initial=1, goal=target_room, rooms_tree=rooms_tree, depth_limit=depth_limit
        )

    def actions(self, state: Any) -> List[Any]:
        node = self._find_node(self.rooms_tree, state)
        actions = []

        if node.left:  # type:ignore
            actions.append(node.left.value)  # type:ignore
        if node.right:  # type:ignore
            actions.append(node.right.value)  # type:ignore

        return actions

    def result(self, state: Any, action: Any) -> Any:
        return action

    def is_goal(self, state: Any) -> bool:
        return state == self.goal  # type:ignore

    def _find_node(self, node: Any, value: Any) -> Optional[Any]:
        if not node:
            return None

        if node.value == value:
            return node

        left_result = self._find_node(node.left, value)
        if left_result:
            return left_result

        right_result = self._find_node(node.right, value)
        if right_result:
            return right_result

        return None


class BinaryTreeNode:
    def __init__(
        self,
        value: int,
        left: Optional["BinaryTreeNode"] = None,
        right: Optional["BinaryTreeNode"] = None,
    ) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"<{self.value}>"


def main() -> None:
    rooms_tree = BinaryTreeNode(
        1,
        BinaryTreeNode(2, None, BinaryTreeNode(4)),
        BinaryTreeNode(3, BinaryTreeNode(5), None),
    )

    goal = 4
    limit = 2

    nav_problem = RoombaNavProblem(rooms_tree, goal, limit)

    result = depth_limited_search(nav_problem, limit=limit)

    found_on_depth = result != Node.failure and result != Node.cutoff
    print(f"Найден на глубине: {found_on_depth}")


if __name__ == "__main__":
    main()
