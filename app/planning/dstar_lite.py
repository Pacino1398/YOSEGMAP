from __future__ import annotations

import heapq
import math

__all__ = ["DStarLite"]


class DStarLite:
    def __init__(
        self,
        start,
        goal,
        obs,
        x_max=50,
        y_max=30,
        inflation_radius=1,
        penalty_weight=2.0,
        passable_obs=None,
        terrain_penalties=None,
    ):
        self.start = start
        self.goal = goal
        self.passable_obs = set(passable_obs or ())
        self.obs = set(obs) - self.passable_obs
        self.x_max = x_max
        self.y_max = y_max
        self.inflation_radius = inflation_radius
        self.penalty_weight = penalty_weight
        self.penalty_map = {}
        self.terrain_penalties = dict(terrain_penalties or {})
        self.g = {}
        self.rhs = {}
        self.U = {}
        self._open_heap = []
        self.km = 0
        self.motions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        for x in range(x_max):
            for y in range(y_max):
                self.g[(x, y)] = float("inf")
                self.rhs[(x, y)] = float("inf")

        for obstacle in self.obs:
            self._apply_inflation(obstacle)

        self.rhs[goal] = 0.0
        self._set_open(goal, self.calc_key(goal))

    def _set_open(self, node, key):
        self.U[node] = key
        heapq.heappush(self._open_heap, (key[0], key[1], node))

    def _pop_open(self):
        while self._open_heap:
            k1, k2, node = heapq.heappop(self._open_heap)
            key = self.U.get(node)
            if key is None:
                continue
            if key[0] == k1 and key[1] == k2:
                return node, key
        return None, None

    def _apply_inflation(self, obs_node):
        ox, oy = obs_node
        affected_nodes = set()

        for dx in range(-self.inflation_radius, self.inflation_radius + 1):
            for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy

                if 0 <= nx < self.x_max and 0 <= ny < self.y_max:
                    if (nx, ny) in self.obs:
                        continue

                    dist = math.hypot(dx, dy)
                    if dist <= self.inflation_radius:
                        added_penalty = (self.inflation_radius - dist + 1) * self.penalty_weight
                        current_penalty = self.penalty_map.get((nx, ny), 0.0)
                        self.penalty_map[(nx, ny)] = current_penalty + added_penalty
                        affected_nodes.add((nx, ny))

        return affected_nodes

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def cost(self, a, b):
        if a in self.obs or b in self.obs:
            return float("inf")

        base_dist = math.hypot(a[0] - b[0], a[1] - b[1])
        safety_penalty = self.penalty_map.get(b, 0.0)
        terrain_penalty = self.terrain_penalties.get(b, 0.0)
        return base_dist + safety_penalty + terrain_penalty

    def calc_key(self, s):
        k1 = min(self.g[s], self.rhs[s]) + self.heuristic(self.start, s) + self.km
        k2 = min(self.g[s], self.rhs[s])
        return (k1, k2)

    def neighbors(self, s):
        nlist = []
        for dx, dy in self.motions:
            nx = s[0] + dx
            ny = s[1] + dy
            if 0 <= nx < self.x_max and 0 <= ny < self.y_max:
                nlist.append((nx, ny))
        return nlist

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(self.cost(u, v) + self.g[v] for v in self.neighbors(u))
        if u in self.U:
            del self.U[u]
        if self.g[u] != self.rhs[u]:
            self._set_open(u, self.calc_key(u))

    def compute_path(self):
        while True:
            if not self.U:
                break
            s, k_old = self._pop_open()
            if s is None:
                break
            k_new = self.calc_key(s)

            if k_old >= k_new and self.rhs[s] == self.g[s]:
                break

            del self.U[s]

            if k_old < k_new:
                self._set_open(s, k_new)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for neighbor in self.neighbors(s):
                    self.update_vertex(neighbor)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                for neighbor in self.neighbors(s):
                    self.update_vertex(neighbor)

    def plan(self):
        self.compute_path()
        path = []
        cur = self.start
        for _ in range(1000):
            path.append(cur)
            if cur == self.goal:
                break
            min_cost = float("inf")
            best = None
            for neighbor in self.neighbors(cur):
                cost = self.cost(cur, neighbor) + self.g[neighbor]
                if cost < min_cost:
                    min_cost = cost
                    best = neighbor
            if best is None:
                break
            cur = best
        return path

    def update_start(self, new_start):
        self.km += self.heuristic(self.start, new_start)
        self.start = new_start

    def update_goal(self, new_goal):
        self.goal = new_goal
        self.__init__(
            self.start,
            self.goal,
            self.obs,
            self.x_max,
            self.y_max,
            self.inflation_radius,
            self.penalty_weight,
            self.passable_obs,
            self.terrain_penalties,
        )

    def update_obstacles(self, new_obstacles):
        nodes_to_update = set()

        for obstacle in new_obstacles:
            if obstacle not in self.obs:
                self.obs.add(obstacle)
                nodes_to_update.add(obstacle)
                nodes_to_update.update(self._apply_inflation(obstacle))

        for node in nodes_to_update:
            self.update_vertex(node)
            for neighbor in self.neighbors(node):
                self.update_vertex(neighbor)

        self.compute_path()
