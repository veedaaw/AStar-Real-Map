# ---------------------------------------------------------------------
# Project 1
# Written by Vida Abdollahi 40039052
# For COMP 6721 Section FK – Fall 2019
# ---------------------------------------------------------------------


import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from shapely.geometry import Point
import timeit


#---------------------------------Class Node-----------------------------------#
class Node:

    def __init__(self, xcoord ,ycoord, xgrid, ygrid):
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.x = xgrid
        self.y = ygrid
        self.gcost =0
        self.hcost =0
        self.fcost =0

        self.parent = None
        self.totalpoints = 0
        self.isblocked = False

    def increase_point(self):
        self.totalpoints = self.totalpoints + 1

    def get_totalPoints(self):
        return self.totalpoints

    def set_block(self, bool):
        self.isblocked = bool
    def get_block(self):
        return self.isblocked

    def get_fcost(self):
        return self.gcost + self.hcost

    def set_parent(self,node):
        self.parent = node

    def get_parent(self):
        return self.parent


##---------------------------preprocessing and input-------------------------------##

print("Enter size of the grids: ")
diameter = float(input())
print("Enter the threshold (50, 75, 90): ")
threshold = int(input())
p1_x, p1_y = [float(x) for x in input("Start point coordinate: ").split()]
p2_x, p2_y = [float(x) for x in input("Goal point coordinate: ").split()]

# I have added an extra row and col to the matrix
points: GeoDataFrame = gpd.read_file('crime_dt.shp')
xmin, ymin, xmax, ymax = points.total_bounds
radius = diameter/2
Xmax = xmax + 2 * diameter
Xmin = xmin - 2 * diameter
Ymax = ymax + 2 * diameter
Ymin = ymin - 2 * diameter

cols = int((Xmax - Xmin) / diameter)
rows = int((Ymax - Ymin) / diameter)

# 2D matrix of points, initialized by 0
matrix = np.zeros([cols,rows], dtype = Node)
block_matrix = np.zeros([cols, rows], dtype=float)
density_matrix = np.zeros([cols, rows], dtype=int)

##--------------------------Functions-----------------------------##

# This function generates a matrix of node based on the actual map
def create_matrix():
    for x in range(0,cols,1):
        for y in range(0,rows,1):
            x_coord = Xmin + x * diameter + radius
            y_coord = Ymin + y * diameter + radius
            matrix[x,y] = Node(x_coord,y_coord,x,y)

    for point in points.geometry:
        n = node_from_map(point)
        n.increase_point()

# To clamp values between 0 and 1
def clamp(num, min_value, max_value):
   return float(max(min(num, max_value), min_value))

# given an actual coordination of a point, this function returns its value in matrix
def node_from_map(point):

    xAxisSize= abs(Xmax - Xmin)
    yAxisSize = abs(Ymax - Ymin)

    percentX = abs(Xmin-point.x) /xAxisSize
    percentY = abs(Ymin-point.y)/yAxisSize

    percentX = clamp(percentX,0,1)
    percentY = clamp(percentY,0,1)

    x = int((cols-1) *percentX)
    y = int((rows-1) *percentY)

    return matrix[x,y]

# Returns neighbour of a node in the matrix. A node might have up to 8 neighbour nodes
def get_neighbours(node):
    neighbours = []
    for x in range(-1, 2, 1):
        for y in range(-1, 2, 1):
            if x==0 and y==0:
                continue
            checkX = node.x + x
            checkY = node.y + y

            if checkX >= 0 and checkX < cols and checkY >=0 and checkY < rows:
                neighbours.append(matrix[checkX,checkY])
    return neighbours


def get_density(node):
    return node.get_totalPoints()

# Finding block position in the matrix based on the threshold value
def make_block(threshold):
    create_matrix()
    density_points = []
    for x in range(0, cols, 1):
        for y in range(0, rows, 1):
            density_matrix[x, y] = matrix[x, y].get_totalPoints()
            density_points.append(matrix[x, y])

    density_points.sort(reverse=True, key=get_density)

    if(threshold == 50):
        for x in range(0,cols,1):
             for y in range(0,rows,1):
                #mean = np.mean(density_matrix)
                average = np.average(density_matrix)

                if(density_matrix[x,y] >= average):
                    matrix[x][y].set_block(True)
                else:
                    matrix[x][y].set_block(False)

    else:
        index = int( len(density_points) * ((100-threshold)/100))
        for x in range (0, index+1,1):
            density_points[x].set_block(True)
            matrix[density_points[x].x, density_points[x].y].set_block(True)

    for x in range(0, cols, 1):
        for y in range(0, rows, 1):
            if matrix[x][y].get_block() == True:
                block_matrix[x][y] = 1
            if matrix[x][y].get_block() == False:
                block_matrix[x][y] = 0

# Implementation of A* Algorithm
def astar(startPos, targetPos):

    startNode = node_from_map(startPos)
    targetNode = node_from_map(targetPos)

    open_set =[]
    closed_set =[]
    open_set.append(startNode)

    while len(open_set)>0:
        current_node = open_set[0]
        for i in range(1,len(open_set),1):
            if open_set[i].fcost < current_node.fcost or \
                    open_set[i].fcost == current_node.fcost and\
                    open_set[i].hcost < current_node.hcost:
                current_node = open_set[i]
        open_set.remove(current_node)
        closed_set.append(current_node)
        if current_node == targetNode:
            return trace_path(startNode, targetNode)

        for neighbour in get_neighbours(current_node):
            if neighbour.isblocked is True or neighbour in closed_set:
                continue
            cost_to_neighbour = current_node.gcost + get_distance(current_node,neighbour)
            if cost_to_neighbour < neighbour.gcost or neighbour not in open_set:
                neighbour.gcost = cost_to_neighbour
                neighbour.hcost = get_distance(neighbour,targetNode)
                neighbour.set_parent(current_node)
                if neighbour not in open_set:
                    open_set.append(neighbour)

# Returns distance between each two nodes in the matrix_ We can move both in the diagonal and horizontal direction with different cost
def get_distance(node_a, node_b):
    distX = abs(node_a.x - node_b.x)
    distY = abs(node_a.y - node_b.y)
    if distX > distY:
        return 14 *distY + 10 *(distX-distY)
    else:
        return 14 *distX + 10 *(distY-distX)

# Returns the path found by A star algorithm
def trace_path(start_node, goal_node):
    path = []
    current_node = goal_node
    while current_node != start_node:
        path.append(current_node)
        current_node = current_node.get_parent()
    return path

##-----------------------------Main------------------------------##


# region Main

make_block(threshold)

point1 = Point(p1_x, p1_y)
point2 = Point(p2_x, p2_y)

#point1 = Point(-73.55, 45.49)
#point2 = Point(-73.59, 45.53)

print("Mean/Average is: ", np.average(density_matrix))
print("Standard Deviation is: ", np.std(density_matrix))

start = timeit.default_timer()
path = astar(point1, point2)
stop = timeit.default_timer()
time = stop - start

if path != None and time <=10:
    for p in path:
        block_matrix[p.x,p.y] = 0.5
        print("Path: ", p.x, p.y)
else:
    print("Due to blocks, no path is found. Please change the map and try again")

print('Time: ', time)
print("Terminated")



##--------------------------PlOT--------------------------##

axis = points.plot()
axis.grid('True')
axis.set_yticks(np.arange(Ymin, Ymax, diameter))
axis.set_xticks(np.arange(Xmin, Xmax, diameter))
plt.title("Original Map of Locations")
plt.show()

plt.matshow(density_matrix)
plt.title("Total Number/Density Matrix")
plt.colorbar()
plt.show()

plt.pcolormesh(block_matrix, edgecolors='k', linewidth=1)
plt.title("A* Search")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

# endregion Main




