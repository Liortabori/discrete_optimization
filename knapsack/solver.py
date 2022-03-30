#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    #sorting items based on value-per-weight
    items.sort(key = lambda x: x[1]/x[2], reverse=True)
    #initialize variables
    taken = [0]*len(items)
    current_value = 0
    best_so_far = 0

    #go into search
    value, taken = dfs(items, capacity, current_value, taken, best_so_far)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def dfs(items, remaining_capacity, current_value, taken, best_value_so_far):
    # depth first search approach

    #initialize current decision
    decision = taken
    #remove non relevant items
    items = [i for i in items if i.weight <= remaining_capacity]

    #pop first item for use/no use in this level
    if len(items) > 0:
        #calculate potential
        potential = calculate_potential(items, remaining_capacity, current_value)
        item = items.pop(0)
    else:
        return current_value, decision

    #explore next level
    for action in ['take', 'dont take']:
        if potential > best_value_so_far:
            new_value,new_capacity,new_taken = inner_loop(item, remaining_capacity, current_value, taken, action)
            final_value, final_taken = dfs(items,new_capacity,new_value,new_taken,best_value_so_far)
            if final_value > best_value_so_far:
                best_value_so_far = final_value
                decision = final_taken
        else:
            pass

    return best_value_so_far, decision

def inner_loop(item, capacity, value, taken, action):
    new_taken = taken.copy()
    if action == 'take':
        #take branch
        value += item.value
        capacity -= item.weight
        new_taken[item.index] = 1
    elif action == 'dont take':
        #don't take branch - don't need to implement anything
        pass
    else:
        raise Exception(f"Sorry, no such action as {action}")

    return value, capacity, new_taken

def calculate_potential(items, remaining_capacity,current_value):
    potential = current_value
    for item in items:
        if item.weight <= remaining_capacity:
            remaining_capacity -= item.weight
            potential += item.value
        else:
            potential += item.value * (remaining_capacity / item.weight)
            break

    return potential

if __name__ == '__main__':
    import sys
    flag = False
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    elif flag:
        file_location = './data/ks_4_0'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

# for item in items:
#     if weight + item.weight <= capacity:
#         taken[item.index] = 1
#         value += item.value
#         weight += item.weight