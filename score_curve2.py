#
# Draw score curve
#

from numpy import *
import pylab as plt
import sys


argvs = sys.argv

score_list = []
for line in open("zoningplanner/zone_mcmc_scores_4x4.txt", 'r'):
	score_list.append(float(line))
plt.plot(xrange(len(score_list)), score_list, label="4x4")

score_list = []
for line in open("zoningplanner/zone_mcmc_scores_8x8.txt", 'r'):
	score_list.append(float(line))
plt.plot(xrange(200000, 200000+len(score_list)), score_list, label="8x8")

score_list = []
for line in open("zoningplanner/zone_mcmc_scores_16x16.txt", 'r'):
	score_list.append(float(line))
plt.plot(xrange(400000, 400000+len(score_list)), score_list, label="16x16")

score_list = []
for line in open("zoningplanner/zone_mcmc_scores_32x32.txt", 'r'):
	score_list.append(float(line))
plt.plot(xrange(600000, 600000+len(score_list)), score_list, label="32x32")


plt.xlabel("MCMC steps");
plt.ylabel("score");
plt.ylim(-1, 0.5)
plt.legend(loc="lower left")

plt.savefig("score_evolution_curve.png")
plt.show()