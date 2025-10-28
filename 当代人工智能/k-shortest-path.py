import heapq


def astar(graph, start, end, H):
    open_set = [(H[start], start, [])]  # 优先级队列，(f, node, path)
    k_shortest_paths = []  # 存储K短路路径

    while open_set:
        f, current, path = heapq.heappop(open_set)

        if current == end:
            k_shortest_paths.append((f, path + [current]))

        if len(k_shortest_paths) == K:
            break

        if current in graph:
            for neighbor, cost in graph[current]:
                h = H[current]
                g = f - h + cost  # 重新计算路径长度
                h = H[neighbor]  # 启发式估计函数值
                heapq.heappush(open_set, (g + h, neighbor, path + [current]))
        else:
            continue

    return k_shortest_paths


def dijkstra(graph, end):
    # 使用Dijkstra算法计算从start到end的最短距离
    # distances = {node: float('inf') for node in graph}
    distances = {node: float('inf') for node in range(1,end+1)}

    distances[end] = 0
    priority_queue = [(0, end)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        if current_node in graph:
            for neighbor, weight in graph[current_node]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        else:
            continue

    return distances


if __name__ == '__main__':
    N, M, K = map(int, input().split())
    graph = {}  # 图的数据结构，以邻接列表表示
    graph2 = {}
    for _ in range(M):
        X, Y, D = map(int, input().split())
        if X not in graph:
            graph[X] = []
        graph[X].append((Y, D))
        if Y not in graph2:
            graph2[Y] = []
        graph2[Y].append((X, D))

    start = 1  # 起点
    end = N  # 终点

    H = dijkstra(graph2, end)
    k_shortest_paths = astar(graph, start, end, H)

    for i in range(K):
        if i <len(k_shortest_paths):
            f, path=k_shortest_paths[i]
            print(f)
        else:
            print(-1)
