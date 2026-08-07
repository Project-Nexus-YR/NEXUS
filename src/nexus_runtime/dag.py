"""Dynamic, validated directed acyclic task graph."""

from __future__ import annotations

from collections import defaultdict

from .models import DomainError, Task, TaskState


class TaskDAG:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._children: dict[str, set[str]] = defaultdict(set)

    @property
    def tasks(self) -> dict[str, Task]:
        return dict(self._tasks)

    def add(self, task: Task) -> None:
        if task.task_id in self._tasks:
            raise DomainError(f"task already exists: {task.task_id}")
        missing = task.dependencies - self._tasks.keys()
        if missing:
            raise DomainError(f"task has missing dependencies: {sorted(missing)}")
        if task.task_id in task.dependencies:
            raise DomainError("task cannot depend on itself")
        self._tasks[task.task_id] = task
        for dependency in task.dependencies:
            self._children[dependency].add(task.task_id)
        try:
            self.validate()
        except Exception:
            del self._tasks[task.task_id]
            for dependency in task.dependencies:
                self._children[dependency].discard(task.task_id)
            raise

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        if task_id not in self._tasks or dependency_id not in self._tasks:
            raise DomainError("both tasks must exist before adding an edge")
        task = self._tasks[task_id]
        if task.state not in {TaskState.CREATED, TaskState.READY}:
            raise DomainError("dependencies cannot change after a task is leased")
        task.dependencies.add(dependency_id)
        self._children[dependency_id].add(task_id)
        try:
            self.validate()
        except Exception:
            task.dependencies.remove(dependency_id)
            self._children[dependency_id].discard(task_id)
            raise

    def validate(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise DomainError(f"dependency cycle includes {task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self._tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self._tasks:
            visit(task_id)

    def ready(self) -> list[Task]:
        return [
            task
            for task in self._tasks.values()
            if task.state in {TaskState.CREATED, TaskState.RETRYING}
            and all(
                self._tasks[dependency].state == TaskState.COMPLETED
                for dependency in task.dependencies
            )
        ]

    def descendants(self, task_id: str) -> set[str]:
        seen: set[str] = set()
        pending = list(self._children[task_id])
        while pending:
            candidate = pending.pop()
            if candidate not in seen:
                seen.add(candidate)
                pending.extend(self._children[candidate])
        return seen
