#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import namedtuple

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    solution = range(node_count)
    #go into the calculation
    solution = find_minimal_coloring(node_count,edges, solution)

    # prepare the solution in the specified output format
    output_data = str(max(solution) + 1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def find_minimal_coloring(node_count, edges, solution):
    upper_bound = node_count
    tmp = [np.nan] * node_count
    cs = constraint_store(node_count, edges, upper_bound)
    new_edges = edges.copy()
    counter = 0
    #first approximation
    while np.nan in tmp:
        if counter % 100 == 0:
            #prune edges
            new_edges = prune_edges(new_edges, tmp)
            #order remaining open vertexes
            order_list = choose_next_point_order(tmp, cs)
            print(f'we still have {len(order_list)} points to color')
        #choose a point
        v = order_list.pop(0)
        #color it
        tmp = color_vertex(v, tmp, upper_bound, cs, new_edges)
        counter += 1

    if not np.nan in tmp:
        solution = tmp.copy()
        upper_bound = len(set(solution))
        cs._update_upper_bound(upper_bound)
        print(f'upper_bound found is: {upper_bound}')
        # print(solution)

    #outer loop
    while True:
        better_solution = replace_color_ordering(solution,upper_bound,edges,cs)
        if max(better_solution) < max(solution):
            solution = better_solution.copy()
            upper_bound = len(set(solution))
            cs._update_upper_bound(upper_bound)
            print(f'upper_bound found is: {upper_bound}')
        else:
            break

    return solution

def replace_color_ordering(solution,upper_bound,edges,cs):
    new_solution = solution.copy()
    vertexes_to_consider = [v for (v,s) in enumerate(new_solution) if s == upper_bound -1]
    while len(vertexes_to_consider) > 0:
        explore_que = []
        #check if there is a theoretical solution. if not break
        #first lets find a path
        vertex = vertexes_to_consider.pop()
        print(f'vertex is: {vertex}')
        path = find_possible_paths(vertex, cs, edges, new_solution, explore_que, 0, 1, [])
        for p in path:
            #now lets explore a new solution after swapping some colors
            print(f'swapping {p[0]} and {p[1]}')
            new_solution = swapping(new_solution, cs, p)
        if check_feasibility(edges, new_solution):
            if max(new_solution) < max(solution):
                solution = new_solution.copy()
                return solution
            elif new_solution != solution:
                pass
        else:
            break

    return solution

def find_possible_paths(vertex, cs, edges, new_solution, explore_que, path_cost = 0, level = 1, potential_paths = []):
    if level >= 10:
        return []
    open_vertexes =  [v for v in range(len(new_solution)) if cs.find_open_vertex(v)]
    relevant_edges = [(x,y) for x,y in edges if (x==vertex or y==vertex)]
    neighbors = [item for sublist in relevant_edges for item in sublist if item != vertex]
    colors_constraints = {}
    for c in range(max(new_solution)):
        colors_constraints[c] = [n for n in neighbors if new_solution[n] == c]

    possible_changes = {}
    for i in [n for n in neighbors if n in open_vertexes]:
        possible_changes[new_solution[i]] = possible_changes.get(new_solution[i], [])
        possible_changes[new_solution[i]].append(i)

    for c in range(max(new_solution)):
        try:
            counter = len(possible_changes[c]) - len(colors_constraints[c])
            if counter >= 0:
                explore_que.append((level, True, -path_cost, potential_paths + [(s,vertex) for s in possible_changes[c]], [], new_solution[possible_changes[c][0]]))
            else:
                unsolved = [x for x in colors_constraints[c] if x not in possible_changes[c]]
                explore_que.append((level, False, -(path_cost + len(unsolved) * level), potential_paths + [(s,vertex) for s in possible_changes[c]] , unsolved, new_solution[possible_changes[c][0]]))
        except:
            pass

    explore_que.sort(key = lambda x: (x[1],-x[0], x[2], -len(x[3]), -x[5]),reverse=True)
    if len([flag for (_,flag,_,_,_, _) in explore_que if flag]) > 0:
        potential_paths =  [x for (_,flag,_,x,_,_) in explore_que if flag][0]
    elif len(explore_que) > 0:
        level,_,new_cost,swapped, vertexes_to_explore, color = explore_que.pop(0)
        potential_paths = [find_possible_paths(v, cs, edges, new_solution, explore_que, -new_cost, level+1, swapped) for v in vertexes_to_explore][0]
    else:
        potential_paths =[]
    return potential_paths

def swapping(new_solution, cs, path):
    #swap colors
    destination, origin = path
    chosen_color = new_solution[destination]
    removed_color = new_solution[origin]
    open_color = min([x for x in range(cs.upper_bound) if len(cs.get_single_constraint(destination,x)) == 0])
    new_solution[destination] = open_color
    new_solution[origin] = chosen_color
    #update constraints
    cs._change_constraints(origin,chosen_color, [])
    cs._change_constraints(origin,removed_color, [], 'remove')

    cs._change_constraints(destination,open_color, [])
    cs._change_constraints(destination,chosen_color, [], 'remove')

    return new_solution

def prune_edges(edges, solution):
    new_edges = edges.copy()
    for (e1,e2) in new_edges:
        if not solution[e1] is np.nan and not solution[e2] is np.nan:
            new_edges.remove((e1,e2))
    return new_edges

def choose_next_point_order(solution, cs):
    #order by open edges and then number of constraints. #TODO
    list_of_remaining_vertex = [v for v,s in enumerate(solution) if (s is np.nan)]
    order_list = [(cs.get_detailed_constraints(v),cs._get_open_edges(v), v) for v in list_of_remaining_vertex]
    order_list.sort(key = lambda x: (x[0], x[1]), reverse=True)

    return [v for _,_,v in order_list]

def color_vertex(vertex, solution, upper_bound, cs, edges):
    color = min([x for x in range(upper_bound) if len(cs.get_single_constraint(vertex,x)) == 0])
    solution[vertex] = color
    cs._change_constraints(vertex, color, edges)

    return solution

class constraint_store:
    #keep up with current constraints
    def __init__(self,num_vertices,edges, upper_bound):
        self.edges = edges
        self.edges_constraints = {}
        self.upper_bound = upper_bound
        self.num_vertices = num_vertices
        self.num_open_edges = {}
        for v in range(num_vertices):
            self.edges_constraints[v] = {}
            for x in range(upper_bound):
                self.edges_constraints[v][x] = []
        for e in edges:
            for i in e:
                self.num_open_edges[i] = self.num_open_edges.get(i,0) + 1

    def find_open_vertex(self,v):
        for k in range(self.upper_bound - 1):
            if len(self.get_single_constraint(v,k)) == 0:
                return True
        return False

    def get_single_constraint(self,v,k):
        return self.edges_constraints[v][k]

    def _get_constraints(self, v):
        return self.edges_constraints[v]

    def get_detailed_constraints(self,v):
        ans = 0
        for k in range(self.upper_bound):
            ans += len(self.get_single_constraint(v,k)) == 0
        return ans

    def _get_open_edges(self, v):
        return self.num_open_edges[v]

    def _change_single_constraint(self, v, color, vertex, action = 'add'):
        if action == 'add':
            self.edges_constraints[v][color].append(vertex)
        elif action == 'remove':
            self.edges_constraints[v][color].remove(vertex)
        else:
            raise Exception(f"unknown action {action}")

    def _change_constraints(self, v, color, edges, action = 'add'):
        if len(edges) == 0:
            edges = self.edges
        self._change_single_constraint(v, color,v,action)
        for (x,y) in edges:
            if v == x:
                self._change_single_constraint(y, color,x,action)
            elif v == y:
                self._change_single_constraint(x, color,y,action)

    def _update_upper_bound(self, upper_bound):
        for k in self.edges_constraints.keys():
            for i in range(upper_bound, self.upper_bound):
                del self.edges_constraints[k][i]
        self.upper_bound = upper_bound

def check_feasibility(edges, solution):
    for (x,y) in edges:
        if solution[x] == solution[y]:
            print(f'problem in edge {x,y}')
            return False
    return True

if __name__ == '__main__':
    import sys
    flag = True
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    elif flag:
        file_location = './data/gc_50_3'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
