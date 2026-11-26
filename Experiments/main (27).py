import numpy as np
import matplotlib.pyplot as plt
marks=np.array([[88,76,91],[72,84,79],[95,90,93],[65,70,68],[81,86,85]])
subjects=["Math","Physics","Computer Science"]
avg=marks.mean(axis=1)
print("Marks (rows=students):\n",marks)
print("Total marks:",marks.sum(axis=1))
print("Average marks:",avg)
print("Highest average: Student",int(avg.argmax()+1))
print("Math x Physics:",marks[:,0]*marks[:,1])
print("Subject averages:",marks.mean(axis=0))
plt.bar(subjects,marks.mean(axis=0));plt.ylabel("Average mark");plt.title("Subject-wise average marks");plt.tight_layout();plt.savefig("subject_averages.png")