import os
import numpy as np
import matplotlib.pyplot as plt

list = os.listdir('../ZoningPlanner/zoning')

data = []

for filename in list:
	print(filename)
	start = filename.find('score_')
	end = filename.find('.xml')
	score = float(filename[start+6:end])
	data.append(score)
	
	
plt.hist(data, bins=40, normed=True)
plt.show()
