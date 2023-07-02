def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]

    state = goal_state
    while prev[state.to_string()] is not None:
        for action in state.get_actions():
            if state.apply_action(action).is_same(prev[state.to_string()]):
                break

        result.append((state, action))
        state = prev[state.to_string()]


    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    # for current_state, action in plan:
    #     print(current_state.to_string())
    #     if action is not None:
    #         print('apply action {}'.format(action))
