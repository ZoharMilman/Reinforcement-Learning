from puzzle import *
from planning_utils import *
import heapq as hq
import datetime
import numpy as np

from rl_hw_1_public.planning_utils import traverse


def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    alpha = 1

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = alpha * initial.get_manhattan_distance(goal)
    # initial_to_goal_heuristic = initial.get_classification_distance(goal)
    # initial_to_goal_heuristic = 10000000000000000000000000000000

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}
    counter = 1;
    while len(fringe) > 0:

        current_dist, current = hq.heappop(fringe)
        if current.to_string() in concluded:
            continue
        else:
            concluded.add(current.to_string())

        actions = current.get_actions()
        edge_states = [current.apply_action(a) for a in actions]

        # Choosing the current state
        for i in range(len(edge_states)):
            if edge_states[i].to_string() not in distances.keys():
                distances[edge_states[i].to_string()] = np.inf
            if distances[edge_states[i].to_string()] > 1 + current_dist:
                distances[edge_states[i].to_string()] = 1 + current_dist
                prev[edge_states[i].to_string()] = current
                hq.heappush(fringe, (distances[edge_states[i].to_string()] + alpha * edge_states[i].get_manhattan_distance(goal), edge_states[i]))
                # hq.heappush(fringe, (distances[edge_states[i].to_string()] + edge_states[i].get_classification_distance(goal), edge_states[i]))
                # hq.heappush(fringe, (
                # distances[edge_states[i].to_string()] + 10000000000000000000000000000000, edge_states[i]))

                counter = counter+1

        if (current == goal): break

    print('counter is ' + str(counter))
    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u',  # 'r', 'd', 'l', 'l', 'd', 'r'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
