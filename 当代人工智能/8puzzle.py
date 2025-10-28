# def heuristic(target, state):
#     return sum(1 for a, b in zip(state, target) if a != b)

def heuristic(target,state):
    dist = 0
    for i in range(1, 9):
        current_row, current_col = state.index(str(i)) // 3, state.index(str(i)) % 3
        target_row, target_col = target.index(str(i)) // 3, target.index(str(i)) % 3
        dist += abs(current_row - target_row) + abs(current_col - target_col)
    return dist


def a_star_search(target, initial_state):
    open_list = [(initial_state, 0)]
    closed_set = set()

    while open_list:
        state, cost = open_list.pop(0)
        if state == target:
            return cost
        if state not in closed_set:
            closed_set.add(state)
            zero_idx = state.index("0")
            neighbors = [(zero_idx - 1, zero_idx), (zero_idx + 1, zero_idx),
                         (zero_idx - 3, zero_idx), (zero_idx + 3, zero_idx)]
            for a, b in neighbors:
                if 0 <= a < 9 and 0 <= b < 9:
                    new_state = list(state)
                    new_state[a], new_state[b] = new_state[b], new_state[a]
                    new_state = "".join(new_state)
                    open_list.append((new_state, cost + 1))
        open_list.sort(key=lambda x: x[1] + heuristic(target, x[0]))
    return -1


target = "135702684"
initial_state = input().strip()
moves = a_star_search(target, initial_state)
print(moves)
