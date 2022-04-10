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

    solution = range(node_count)
    #go into the calculation
    solution = find_minimal_coloring(node_count,edges, solution)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def find_minimal_coloring(node_count, edges, solution):
    upper_bound = node_count
    memory_mechanism = {}
    tmp = [np.nan] * node_count
    cs = constraint_store(node_count, edges, upper_bound)
    #inner loop - try to find a solution. if fails update trajectory
    tmp = first_approximation(edges, tmp, cs, upper_bound)
    if not np.nan in tmp:
        solution = tmp.copy()
        upper_bound = len(set(solution))
        cs._update_upper_bound(upper_bound)
        print(f'upper_bound found is: {upper_bound}')

    #outer loop
    while True:
        better_solution = replace_color_ordering(solution,upper_bound,edges,cs,memory_mechanism)
        if max(better_solution) < max(solution):
            solution = better_solution.copy()
            upper_bound = len(set(solution))
            cs._update_upper_bound(upper_bound)
            print(f'upper_bound found is: {upper_bound}')
        else:
            break
    #
    #
    #         print(check_feasibility(edges, solution))
    #     else:
    #         break

    #divide and conquer with strongly connected components

    #try to color with N-1 colors. fail fast!
    #if we can't reduce number of colors return results (max time to run is 5 hours)
    #break simetry

    return solution

def replace_color_ordering(solution,upper_bound,edges,cs,memory_mechanism):
    vertexes_to_consider = [v for (v,s) in enumerate(solution) if s == upper_bound -1]
    open_vertexes = [v for v in range(len(solution)) if len(cs.edges_constraints[v]) > 0]
    #check if there is a theoretical solution. if not break
    if len(vertexes_to_consider) > len(open_vertexes):
        return solution
    fixed = 0
    new_solution = solution.copy()
    while fixed < len(vertexes_to_consider):
        #TODO insert memory mech
        vertex_num = vertexes_to_consider[fixed] #TODO find smarter ways to order this list
        relevant_edges = [(x,y) for x,y in edges if (x==vertex_num or y==vertex_num)]
        neighbors = set([item for sublist in relevant_edges for item in sublist if item != vertex_num])
        new_solution = replace_color(vertex_num, neighbors, new_solution, cs)
        if new_solution != solution and check_feasibility(edges, new_solution):
            fixed += 1
            solution = new_solution.copy()
        else:
            print(1)
            break
    return solution

def replace_color(vertex, neighbors, new_solution, cs):
    neighbors_list = [(n, cs.edges_constraints[n]) for n in neighbors]
    neighbors_list.sort(key = lambda x: len(x[1])) #TODO secondary metric (lowest color and\or number of constraints)
    while len(neighbors_list) > 0:
        n,values = neighbors_list.pop()
        if len(values) > 0:
            chosen_color = min(values)
            new_solution[vertex] = new_solution[n]
            new_solution[n] = chosen_color
            break
        else:
            break #TODO
    return new_solution

def first_approximation(edges, solution, cs, upper_bound):
    #prune edges
    new_edges = prune_edges(edges, solution)
    #order remaining open vertexes
    order_list = choose_next_point_order(solution, cs)
    #brute force all options #TODO

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

# def update_memory(memory_mechanism, new_edges, v):
#     num_edges_remaining = len(new_edges)
#     decision_vertex = v
#     memory_mechanism[(num_edges_remaining)] = memory_mechanism.get((num_edges_remaining), [])
#     memory_mechanism[(num_edges_remaining)].append(decision_vertex)
#     return memory_mechanism

def choose_next_point_order(solution, cs):
    #order by open edges and then number of constraints. #TODO
    list_of_remaining_vertex = [v for v,s in enumerate(solution) if (s is np.nan)]
    order_list = [(len(cs._get_constraints(v)), v) for v in list_of_remaining_vertex]
    order_list.sort(key = lambda x: (x[0]))

    return order_list

def color_vertex(vertex, solution, upper_bound, cs, edges, action ='remove'):
    if action == 'remove':
        color = min([x for x in range(upper_bound) if x in cs._get_constraints(vertex)]) #TODO
        solution[vertex] = color
        cs._change_constraints(vertex, color, edges, action)

    # elif action == "add": #TODO
    #     if solution[vertex] == max_color - 1:
    #         solution[vertex] = np.nan
    #         cs._change_constraints(vertex, max_color - 1, edges, "remove")
    #         max_color = max(0, len(set([x for x in solution if not x is np.nan])))
    #     else:
    #         pass #solution[vertex] = np.nan
    else:
        raise Exception(f"unfamiliar action {action}")

    return solution

class constraint_store:
    #keep up with current constraints
    def __init__(self,num_vertices,edges, upper_bound):
        self.edges = edges
        self.edges_constraints = {}
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

    def _change_single_constraint(self, v, color, action):
        if action == 'add':
            self.edges_constraints[v].append(color)
        elif action == 'remove':
            self.edges_constraints[v] = [c for c in self.edges_constraints[v] if c != color]
        else:
            raise Exception(f"unfamiliar action {action}")

    def _change_constraints(self, v, color, edges, action = 'remove'):
        self._change_single_constraint(v, color, action)
        for (x,y) in edges:
            if v == x:
                self._change_single_constraint(y, color, action)
            elif v == y:
                self._change_single_constraint(x, color, action)

    def _update_upper_bound(self, upper_bound):
        for k,v in self.edges_constraints.items():
            self.edges_constraints[k] = [c for c in self.edges_constraints[k] if c < upper_bound - 1]

def check_feasibility(edges, solution):
    for (x,y) in edges:
        if solution[x] == solution[y]:
            print(x,y)
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
        file_location = './data/gc_50_1'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')


        # #update memory
        # memory_mechanism = update_memory(memory_mechanism, new_edges, v)