
import re
import sys
import numpy as np
import math



def main():
    alg = sys.argv[1]

    with open(sys.argv[2]) as f:
        lines = f.read().splitlines()
        if len(lines) == 2 or len(lines) == 3:
            start = lines[0]
            goal = lines[1]
            forbids = []
            if len(lines) == 3:
                forbids = lines[2].split(",")
        else:
            print("Invalid input.")
            sys.exit()
    
    start_idx = convert_input(start)
    goal_idx = convert_input(goal)
    forbidden = [convert_input(i) for i in forbids]

    if alg == "B":
        return bfs(start_idx, goal_idx, forbidden)
    elif alg == "D":
        return dfs(start_idx, goal_idx, forbidden)
    elif alg == "I":
        return ids(start_idx, goal_idx, forbidden)
    elif alg == "G":
        return greedy(start_idx, goal_idx, forbidden)
    elif alg == "A":
        return a_star(start_idx, goal_idx, forbidden)
    elif alg == "H":
        return hill(start_idx, goal_idx, forbidden)
    else:
        print("Search algorithm not defined.")


def convert_input(s):
    return [int(i) for i in s]


def print_answer(nums):
    print(','.join(list(map(lambda idx: ''.join(map(str, idx)), nums))))


# BFS    
def bfs(start_idx, goal_idx, forbidden):
    queue = [(start_idx, [], [0,0,0])] # current_idx, path, last_move
    visited = []
 
    while queue:
        current_idx, path, last_move = queue.pop(0) # remove first element from queue
        visited.append((current_idx, last_move)) # record expanded nodes
 
        if len(visited) >= 1000:
            print("No solution found.") 
            print_answer(list(map(lambda p: p[0], visited))) # gets path element from visited tuple
            return
 
        path.append(current_idx)
 
        if current_idx == goal_idx:
            print_answer(path)
            print_answer(list(map(lambda p: p[0], visited)))
            return
 
        for i in range(3):
            if last_move[i] == 1:
                continue
            move = [0, 0, 0]
            move[i] = 1
            n = current_idx[:]
            n[i] -=1 
            if n[i] >= 0 and n not in forbidden and (n, move) not in visited:
                queue.append((n, path[:], move))
            n = current_idx[:]
            n[i] += 1
            if n[i] <= 9 and n not in forbidden and (n, move) not in visited:
                queue.append((n, path[:], move))


#DFS
def dfs(start_idx, goal_idx, forbidden):
    stack = [(start_idx, [], [0,0,0])]
    visited = []

    while stack:
        current_idx, path, last_move = stack.pop() # remove first element from stack
        visited.append((current_idx, last_move)) # record expanded nodes

        if len(visited) >= 1000:
            print("No solution found.") 
            print_answer(list(map(lambda p: p[0], visited))) # gets path element from visited tuple
            return

        path.append(current_idx)

        if current_idx == goal_idx:
            print_answer(path)
            print_answer(list(map(lambda p: p[0], visited)))
            return

        for i in range(2, -1, -1):
            if last_move[i] == 1:
                continue
            move = [0, 0, 0]
            move[i] = 1
            n = current_idx[:]
            n[i] +=1 
            if n[i] <= 9 and n not in forbidden and (n, move) not in visited:
                stack.append((n, path[:], move))
            n = current_idx[:]
            n[i] -= 1
            if n[i] >= 0 and n not in forbidden and (n, move) not in visited:
                stack.append((n, path[:], move))


    #IDS
def ids(start_idx, goal_idx, forbidden):
    visited = []
 
    ids_depth = 0
    while True:
        stack = [(start_idx, [], [0,0,0], 0)]
 
        while stack:
            current_idx, path, last_move, depth = stack.pop() # remove first element from stack
            if depth > ids_depth:
                continue
            visited.append((current_idx, last_move, ids_depth)) # record expanded nodes
 
            if len(visited) >= 1000:
                print("No solution found.") 
                print_answer(list(map(lambda p: p[0], visited))) # gets path element from visited tuple
                return
 
            path.append(current_idx)
 
            if current_idx == goal_idx:
                print_answer(path)
                print_answer(list(map(lambda p: p[0], visited)))
                return
 
            for i in range(2, -1, -1):
                if last_move[i] == 1:
                    continue
                move = [0, 0, 0]
                move[i] = 1
                n = current_idx[:]
                n[i] +=1 
                if n[i] <= 9 and n not in forbidden and (n, move, ids_depth) not in visited:
                    stack.append((n, path[:], move, depth + 1))
                n = current_idx[:]
                n[i] -= 1
                if n[i] >= 0 and n not in forbidden and (n, move, ids_depth) not in visited:
                    stack.append((n, path[:], move, depth + 1))
        
        ids_depth += 1


def heuristic(start,goal):
    if start == None:
        return math.inf
    return sum([abs(a-b) for a,b in zip(start,goal)])


    #Greedy
def greedy(start_idx, goal_idx, forbidden):
    queue = PriorityQueue()
    queue.add((start_idx, [], [0,0,0]), heuristic(start_idx, goal_idx)) 
    visited = []

    while queue.size > 0:
        current_idx, path, last_move = queue.remove_min() # remove first element from queue
        visited.append((current_idx, last_move)) # record expanded nodes

        if len(visited) >= 1000:
            print("No solution found.") 
            print_answer(list(map(lambda p: p[0], visited))) # gets path element from visited tuple
            return

        path.append(current_idx)

        if current_idx == goal_idx:
            print_answer(path)
            print_answer(list(map(lambda p: p[0], visited)))
            return

        for i in range(3):
            if last_move[i] == 1:
                continue
            move = [0, 0, 0]
            move[i] = 1
            n = current_idx[:]
            n[i] -=1 
            if n[i] >= 0 and n not in forbidden and (n, move) not in visited:
                queue.add((n, path[:], move), heuristic(n, goal_idx))
            n = current_idx[:]
            n[i] += 1
            if n[i] <= 9 and n not in forbidden and (n, move) not in visited:
                queue.add((n, path[:], move), heuristic(n, goal_idx))


    #A*
def a_star(start_idx, goal_idx, forbidden):
    queue = PriorityQueue()
    queue.add((start_idx, [], [0,0,0]), heuristic(start_idx, goal_idx)) 
    visited = []

    while queue.size > 0:
        current_idx, path, last_move = queue.remove_min() # remove first element from queue
        visited.append((current_idx, last_move)) # record expanded nodes

        if len(visited) >= 1000:
            print("No solution found.") 
            print_answer(list(map(lambda p: p[0], visited))) # gets path element from visited tuple
            return

        path.append(current_idx)

        if current_idx == goal_idx:
            print_answer(path)
            print_answer(list(map(lambda p: p[0], visited)))
            return

        for i in range(3):
            if last_move[i] == 1:
                continue
            move = [0, 0, 0]
            move[i] = 1
            n = current_idx[:]
            n[i] -=1 
            if n[i] >= 0 and n not in forbidden and (n, move) not in visited:
                queue.add((n, path[:], move), heuristic(n, goal_idx)+len(path))
            n = current_idx[:]
            n[i] += 1
            if n[i] <= 9 and n not in forbidden and (n, move) not in visited:
                queue.add((n, path[:], move), heuristic(n, goal_idx)+len(path))


    #Hill-climbing 
def hill(start_idx, goal_idx, forbidden):
    visited = [start_idx]
    current_idx = start_idx
    last_move = [0,0,0]

    while current_idx != goal_idx:
        if len(visited) >= 1000:
            print("No solution found.") 
            print_answer(visited) # gets path element from visited tuple
            return

        best_idx = current_idx
        best_last_move = last_move

        for i in range(3):
                if last_move[i] == 1:
                    continue
                move = [0, 0, 0]
                move[i] = 1
                n = current_idx[:]
                n[i] -=1 
                if n[i] >= 0 and n not in forbidden and n not in visited and (heuristic(best_idx, goal_idx) >= heuristic(n, goal_idx)):
                    best_idx = n
                    best_last_move = move
                n = current_idx[:]
                n[i] += 1
                if n[i] <= 9 and n not in forbidden and n not in visited and (heuristic(best_idx, goal_idx) >= heuristic(n, goal_idx)):
                    best_idx = n
                    best_last_move = move
                
        if best_idx == current_idx:
            print("No solution found.") 
            print_answer(visited) 
            return
        
        visited.append(best_idx)
        current_idx = best_idx
        last_move = best_last_move

    print_answer(visited)
    print_answer(visited)


class PriorityQueue():
    def __init__(self):
        self.queue = [0 for _ in range(1000)]
        self.size = 0
        self.count = 0
    
    def add(self, val, priority):
        self.queue[self.size] = (val, priority, self.count)
        self.size += 1
        self.count += 1

        self.sift_up(self.size - 1)

    def remove_min(self):
        if self.size == 0:
            return None
        
        minimum = self.queue[0]

        self.size -= 1
        self.queue[0] = self.queue[self.size]
        self.sift_down(0)

        return minimum[0]

    def left_child(self, i):
        return (i * 2) + 1
    
    def right_child(self, i):
        return (i * 2) + 2

    def parent(self, i):
        return (i - 1) // 2

    def has_left_child(self, i):
        return self.left_child(i) < self.size

    def has_right_child(self, i):
        return self.right_child(i) < self.size

    def sift_up(self, i):
        if i == 0:
            return
        p_idx = self.parent(i)
        _, pp, pc = self.queue[p_idx]
        _, cp, cc = self.queue[i]

        if pp < cp or (pp == cp and pc > cc):
            return

        # parent priority higher, or equal and count lower
        
        self.queue[p_idx], self.queue[i] = self.queue[i], self.queue[p_idx]
        self.sift_up(p_idx)

    def sift_down(self, i):
        if not self.has_left_child(i):
            return
        
        l_idx = self.left_child(i)
        _, lp, lc = self.queue[l_idx]
        _, cp, cc = self.queue[i]

        min_c = l_idx
        min_cc = lc
        min_cp = lp

        if self.has_right_child(i):
            r_idx = self.right_child(i)
            _, rp, rc = self.queue[r_idx]

            if rp < lp or (rp == lp and rc > lc):
                min_c = r_idx
                min_cc = rc
                min_cp = rp
        
        if cp < min_cp or (cp == min_cp and cc > min_cc):
            return


        self.queue[min_c], self.queue[i] = self.queue[i], self.queue[min_c]
        self.sift_down(min_c)
        


if __name__ == "__main__":
    main()