import numpy as np
import matplotlib.pyplot as plt
prior=0.01; sensitivity=0.99; false_positive=0.05
posterior=(sensitivity*prior)/(sensitivity*prior+false_positive*(1-prior))
print(f"Prior P(D): {prior:.4f}")
print(f"Sensitivity: {sensitivity:.2f}; false-positive rate: {false_positive:.2f}")
print(f"P(D|positive): {posterior:.4f} ({posterior*100:.2f}%)")
plt.bar(["Prior","Posterior given +"],[prior,posterior]);plt.ylabel("Probability");plt.title("Bayes diagnostic update");plt.ylim(0,1);plt.tight_layout();plt.savefig("bayes_probabilities.png")