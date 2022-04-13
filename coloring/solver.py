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
    #inner loop - try to find a solution. if fails update trajectory
    tmp = first_approximation(edges, tmp, cs, upper_bound)
    if not np.nan in tmp:
        solution = tmp.copy()
        upper_bound = len(set(solution))
        cs._update_upper_bound(upper_bound)
        print(f'upper_bound found is: {upper_bound}')
        print(solution)

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
    while max(new_solution) == upper_bound - 1:
        explore_que = []
        open_vertexes = [v for v in range(len(new_solution)) if len(cs.edges_constraints[v]) > 0]
        #check if there is a theoretical solution. if not break
        if len(vertexes_to_consider) > len(open_vertexes):
            return solution
        #first lets find a path
        vertex = vertexes_to_consider.pop()
        print(f'vertex is: {vertex}')
        path = find_possible_paths(vertex, open_vertexes, edges, new_solution, explore_que)
        print(f"vertex is present in swapping plan: {len([(x,y) for x,y in path if x == vertex or y == vertex]) > 0}")
        for p in path:
            #now lets explore a new solution after swapping some colors
            print(f'swapping {p[0]} and {p[1]}')
            new_solution = swapping(new_solution, cs, p)
        print(f"new solution is feasible: {check_feasibility(edges, new_solution)}")
        if max(new_solution) < max(solution):
            solution = new_solution.copy()
            return solution
        elif new_solution != solution:
            pass
        else:
            break

    return solution

def find_possible_paths(vertex, open_vertexes, edges, new_solution,explore_que, path_cost = 0, que_so_far = [], level = 1):
    if level == 10:
        return []
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
                explore_que.append((True, -path_cost, que_so_far + [(s,vertex) for s in possible_changes[c]], []))
            else:
                unsolved = [x for x in colors_constraints[c] if x not in possible_changes[c]]
                explore_que.append((False, -(path_cost + len(unsolved) * level), [(s,vertex) for s in possible_changes[c]] , unsolved))
        except:
            pass
    explore_que.sort(key = lambda x: (x[0],x[1], -len(x[2])),reverse=True)
    if sum([flag for (flag,_,_,_) in explore_que]) > 0:
        potential_paths =  [x for (flag,_,x,_) in explore_que if flag][0]
    else:
        _,new_cost,swapped, vertexes_to_explore = explore_que.pop(0)
        for v in vertexes_to_explore:
            new_open = [x for x in open_vertexes if not x in swapped]
            potential_paths = find_possible_paths(v, new_open, edges, new_solution, explore_que, -new_cost, swapped, level+1)

    return potential_paths

def swapping(new_solution, cs, path):
    #swap colors
    destination, origin = path
    chosen_color = new_solution[destination]
    open_color = min(cs.edges_constraints[destination])
    new_solution[destination] = open_color
    new_solution[origin] = chosen_color
    #update constraints
    cs._refresh_constraints(origin,new_solution)
    cs._refresh_constraints(destination,new_solution)
    return new_solution

def first_approximation(edges, solution, cs, upper_bound):
    #prune edges
    new_edges = prune_edges(edges, solution)
    #order remaining open vertexes
    order_list = choose_next_point_order(solution, cs)
    #brute force all options

    if len(order_list) == 0:
        return solution
    else:
        #choose a point
        c, v = order_list.pop(0)

        #color it
        solution = color_vertex(v, solution, upper_bound, cs, new_edges)

        # Go deeper
        solution = first_approximation(new_edges, solution, cs, upper_bound)

    return solution


def prune_edges(edges, solution):
    new_edges = edges.copy()
    for (e1,e2) in new_edges:
        if not solution[e1] is np.nan and not solution[e2] is np.nan:
            new_edges.remove((e1,e2))
    return new_edges

def choose_next_point_order(solution, cs):
    #order by open edges and then number of constraints. #TODO
    list_of_remaining_vertex = [v for v,s in enumerate(solution) if (s is np.nan)]
    order_list = [(len(cs._get_constraints(v)), v) for v in list_of_remaining_vertex]
    order_list.sort(key = lambda x: (x[0]))

    return order_list

def color_vertex(vertex, solution, upper_bound, cs, edges):
    color = min([x for x in range(upper_bound) if x in cs._get_constraints(vertex)])
    solution[vertex] = color
    cs._change_constraints(vertex, color, edges)

    return solution

class constraint_store:
    #keep up with current constraints
    def __init__(self,num_vertices,edges, upper_bound):
        self.edges = edges
        self.edges_constraints = {}
        self.upper_bound = upper_bound
        # self.num_open_edges = {}
        for v in range(num_vertices):
            self.edges_constraints[v] = range(upper_bound)
        # for e in edges:
        #     for i in e:
        #         self.num_open_edges[i] = self.num_open_edges.get(i,0) + 1

    def _get_constraints(self, v):
        return self.edges_constraints[v]

    # def _get_open_edges(self, v):
    #     return self.num_open_edges[v]

    def _change_single_constraint(self, v, color):
        self.edges_constraints[v] = [c for c in self.edges_constraints[v] if c != color]

    def _change_constraints(self, v, color, edges):
        self._change_single_constraint(v, color)
        for (x,y) in edges:
            if v == x:
                self._change_single_constraint(y, color)
            elif v == y:
                self._change_single_constraint(x, color)

    def _update_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound
        for k,v in self.edges_constraints.items():
            self.edges_constraints[k] = [c for c in self.edges_constraints[k] if c < upper_bound - 1]

    def _refresh_constraints(self,v,solution):
        self.edges_constraints[v] = range(self.upper_bound)
        self._change_single_constraint(v, solution[v])
        for x,y in self.edges:
            if v == x:
                self._change_single_constraint(x, solution[y])
            elif v == y:
                self._change_single_constraint(y, solution[x])

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
        file_location = './data/gc_250_9'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
