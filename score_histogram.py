#
# Draw score histogram
#

from numpy import *
import pylab as plt
import sys


argvs = sys.argv

score_list = []
for line in open("zoningplanner/zone_exhaustive_scores.txt", 'r'):
	score_list.append(float(line))
plt.hist(score_list, bins=40)
plt.xlabel("score");
plt.savefig("score_histogram.png")
plt.show()

