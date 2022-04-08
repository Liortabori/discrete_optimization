#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import numpy as np

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

    solution = [np.nan] * node_count
    #go into the calculation
    solution = find_minimal_coloring(node_count,edges, solution)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def find_minimal_coloring(node_count, edges, solution):
    lower_bound = 0
    upper_bound = node_count
    memory_mechanism = {}
    # routes_taken = [] #do i need this?

    #outer loop
    while lower_bound < upper_bound:
        max_color = 0
        tmp = [np.nan] * node_count
        cs = constraint_store(node_count, edges)
        #inner loop - try to find a solution. if fails update trajectory
        tmp, flag = forward_pass(edges, tmp, max_color, cs, upper_bound, memory_mechanism)
        if flag:
            if len(set(solution)) < upper_bound:
                solution = tmp.copy()
                upper_bound = len(set(solution))
                print(f'upper_bound found is: {upper_bound}')
                print(solution)
                print(check_feasibility(edges, solution))
        else:
            break

    #divide and conquer with strongly connected components

    #try to color with N-1 colors. fail fast!
    #if we can't reduce number of colors return results (max time to run is 5 hours)
    #break simetry

    return solution

def forward_pass(edges, solution, max_color, cs, upper_bound, memory_mechanism):
    flag = False
    #stop if there is a full solution, otherwise continue search
    if np.nan in solution:
        #prune edges
        new_edges = edges.copy()
        for (e1,e2) in new_edges:
            if not solution[e1] is np.nan and not solution[e2] is np.nan:
                new_edges.remove((e1,e2))
        #order remaining open vertexes
        vertexes = choose_next_point_order(solution, cs, memory_mechanism.get((len(new_edges), max_color),[]))
        #brute force all options #TODO
        while len(vertexes) >= 0:
            if len(vertexes) == 0:
                flag = False
                return solution,flag
            else:
                #choose a point
                v = vertexes.pop(0)
                if v not in memory_mechanism.get((len(new_edges), max_color),[]):
                    #update memory
                    memory_mechanism = update_memory(memory_mechanism, new_edges, max_color, v)

                    #color it
                    if max_color <= upper_bound:
                        solution, max_color = color_vertex(v, solution, max_color, cs, new_edges)
                    else:
                        solution, max_color = color_vertex(v, solution, max_color - 1, cs, new_edges)

                    #check if we passed our best solution. if yes break and update trajectory
                    if max_color > upper_bound:
                        # solution, max_color = color_vertex(v, solution, max_color, cs, new_edges, "remove")
                        return solution, False
                    else:
                        # Go deeper
                        solution, flag = forward_pass(new_edges, solution, max_color, cs, upper_bound, memory_mechanism)
                        if flag:
                            return solution, flag
    else:
        flag = True
        return solution,flag

def update_memory(memory_mechanism, new_edges, max_color, v):
    num_edges_remaining = len(new_edges)
    decision_vertex = v
    memory_mechanism[(num_edges_remaining,max_color)] = memory_mechanism.get((num_edges_remaining,max_color), [])
    memory_mechanism[(num_edges_remaining,max_color)].append(decision_vertex)
    return memory_mechanism

def choose_next_point_order(solution, cs, memory_check_point):
    #order by open edges and then number of constraints. #TODO
    list_of_remaining_vertex = [v for v,s in enumerate(solution) if (s is np.nan) and (v not in memory_check_point)]
    # order_list = [(cs._get_open_edges(v), len(cs._get_constraints(v)), v) for v in list_of_remaining_vertex]
    order_list = [(sum(cs._get_constraints(v)), v) for v in list_of_remaining_vertex]
    order_list.sort(key = lambda x: (x[0]), reverse=True)

    return [v for (constraints, v) in order_list]

def color_vertex(vertex, solution, max_color, cs, edges, action ='add'):
    if action == 'add':
        if solution[vertex] is np.nan:
            color = min([x for x in range(max_color + 1) if not x in cs._get_constraints(vertex)]) #TODO
            solution[vertex] = color
            cs._change_constraints(vertex, color, edges)
            if color == max_color:
                max_color+=1

    elif action == "remove":
        if solution[vertex] == max_color - 1:
            solution[vertex] = np.nan
            cs._change_constraints(vertex, max_color - 1, edges, "remove")
            max_color = max(0, len(set([x for x in solution if not x is np.nan])))
        else:
            pass #solution[vertex] = np.nan
    else:
        raise Exception(f"unfamiliar action {action}")

    return solution, max_color

class constraint_store:
    #keep up with current constraints
    def __init__(self,num_vertices,edges):
        self.edges = edges
        self.edges_constraints = {}
        self.num_open_edges = {}
        self.open_colors = '' #TODO
        for v in range(num_vertices):
            self.edges_constraints[v] = []
        for e in edges:
            for i in e:
                self.num_open_edges[i] = self.num_open_edges.get(i,0) + 1

    def _get_constraints(self, v):
        return self.edges_constraints[v]

    def _get_open_edges(self, v):
        return self.num_open_edges[v]

    def _change_single_constraint(self, v, color, action):
        if action == 'add':
            self.edges_constraints[v].append(color)
        elif action == 'remove':
            self.edges_constraints[v].remove(color)
        else:
            raise Exception(f"unfamiliar action {action}")

    def _change_constraints(self, v, color, edges, action = 'add'):
        for (x,y) in edges:
            if v == x:
                self._change_single_constraint(y, color, action)
            elif v == y:
                self._change_single_constraint(x, color, action)

def check_feasibility(edges, solution):
    for (x,y) in edges:
        if solution[x] == solution[y]:
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
        file_location = './data/gc_100_1'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')