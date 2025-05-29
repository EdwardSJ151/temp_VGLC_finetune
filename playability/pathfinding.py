'''

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
from math import sqrt
from heapq import heappush, heappop

def astar_shortest_path(src, isdst, adj,subOptimal,heuristic):
    dist = {}
    prev = {}
    dist[src] = 0
    prev[src] = None
    heap = [(dist[src], src,0)]

    pathLength = float('inf')
    paths = []
    while heap:
        node = heappop(heap)
        if isdst(node[1]):
            if node[0] < pathLength:
                pathLength = node[0]
                path = []
                nodeR = node[1]
                while nodeR:
                    path.append(nodeR)
                    nodeR = prev[nodeR]
                path.reverse()
                paths.append(path)
                continue
            elif node[0] > pathLength+subOptimal:
                break
            else:
                path = []
                nodeR = node[1]
                while nodeR:
                    path.append(nodeR)
                    nodeR = prev[nodeR]
                path.reverse()
                paths.append(path)
                continue

        for next_node in adj(node):
            next_node[0] += heuristic(next_node[1])
            next_node.append(heuristic(next_node[1]))
            if next_node[1] not in dist or next_node[0] < dist[next_node[1]]:
                #print node[1],next_node[1],heuristic(next_node[1])
                #exit()

                dist[next_node[1]] = next_node[0]
                prev[next_node[1]] = node[1]
                heappush(heap, next_node)

    return paths
