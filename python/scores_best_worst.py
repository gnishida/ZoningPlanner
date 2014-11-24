###################################################
# This program shows histogram of the score distribution 
# for a good plan and a bad plan.

import os
import numpy as np
import matplotlib.pyplot as plt

best_scores = []
for line in open("best-score.txt", "r"):
	best_scores.append(float(line.rstrip()))

worst_scores = []
for line in open("worst-score.txt", "r"):
	worst_scores.append(float(line.rstrip()))

plt.hist(best_scores, bins=30, color="#5F9BFF", normed=True, label="best plan")
plt.hist(worst_scores, bins=30, color="#F8766D", normed=True, label="worst plan")
plt.legend(loc="upper center")
plt.show()
