#
# Draw score curve
#

from numpy import *
import pylab as plt
import sys



x_list = []

time_list = []
for line in open("computation_distance_map.txt", 'r'):
	time_list.append(float(line))
	
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.bar(xrange(1, len(time_list) + 1), time_list, width=0.4, align="center")
plt.title("Distance Map")
plt.xticks([1,2,3,4,5], ["4x4","8x8","16x16","32x32","64x64"])

plt.ylim(0, 1.5);
plt.xlabel("Grid size");
plt.ylabel("computation time [sec]");

time_list = []
for line in open("computation_people_allocation.txt", 'r'):
	time_list.append(float(line))
	
plt.subplot(1,2,2)
plt.bar(xrange(1, len(time_list) + 1), time_list, width=0.4, align="center")
plt.title("People Allocation")
plt.xticks([1,2,3,4,5], ["4x4","8x8","16x16","32x32","64x64"])

plt.ylim(0, 200);
plt.xlabel("Grid size");
plt.ylabel("computation time [sec]");




plt.savefig("computation_time.png")
plt.show()