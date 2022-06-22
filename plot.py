import matplotlib.pyplot as plt
import numpy as np

# Comment all the remaining plots to plot the single plot

## Model Accuracy vs Learning Rate
x_learning_rate = np.array([0.3, 0.2, 0.1, 0.05, 0.01], dtype = np.float64)
y_model_accuracy = np.array([70.289, 75.449, 79.855, 80.398, 73.224], dtype = np.float64)

# Plot the data
plt.plot(x_learning_rate, y_model_accuracy, marker = 'o')
plt.grid()
plt.xlabel("Learning Rate ( 0-1 )")
plt.ylabel("Model Prediction Accuracy ( in % )")
plt.xlim((0.4, 0))
plt.ylim((0, 100))
plt.title("Model Accuracy v/s Learning Rate")

# Save the plotted image
plt.savefig("./plots/accuracy_vs_learningrate.png")


## Overall Devanagari Script Recognition Model Accuracy with epochs
x_epochs = np.array([1, 2, 3, 4, 5], dtype=np.int64)
y_overall_model_accuracy = np.array([80.811, 85.333, 86.847, 87.333, 87.449], dtype=np.float64)

# Plot the data
plt.plot(x_epochs, y_overall_model_accuracy, marker = 'o')
plt.grid()
plt.xlabel("Number of epochs")
plt.ylabel("Model Prediction Accuracy ( in % )")
plt.xlim((0, 6))
plt.ylim((0, 100))
plt.title("Overall Devanagari Script Recognition Model")

# Save the plotted image
plt.savefig("./plots/overallaccuracy_vs_numberofepochs.png")


## Just Devanagari Number Digit Recognition Model Accuracy with epochs
x_epochs = np.array([1, 2, 3, 4, 5], dtype=np.int64)
y_digit_model_accuracy = np.array([97.333, 98.0, 98.333, 98.466, 98.4], dtype=np.float64)

# Plot the data
plt.plot(x_epochs, y_digit_model_accuracy, marker = 'o')
plt.grid()
plt.xlabel("Number of epochs")
plt.ylabel("Model Prediction Accuracy ( in % )")
plt.xlim((0, 6))
plt.ylim((0, 100))
plt.title("Just Devanagari Number Digit Recognition Model")

# Save the plotted image
plt.savefig("./plots/digitaccuracy_vs_numberofepochs.png")



## Just Devanagari Character Recognition Model Accuracy with epochs
x_epochs = np.array([1, 2, 3, 4, 5], dtype=np.int64)
y_character_model_accuracy = np.array([79.194, 84.203, 86.759, 87.425, 87.861], dtype=np.float64)

# Plot the data
plt.plot(x_epochs, y_character_model_accuracy, marker = 'o')
plt.grid()
plt.xlabel("Number of epochs")
plt.ylabel("Model Prediction Accuracy ( in % )")
plt.xlim((0, 6))
plt.ylim((0, 100))
plt.title("Just Devanagari Character Recognition Model")

# Save the plotted image
plt.savefig("./plots/characteraccuracy_vs_numberofepochs.png")