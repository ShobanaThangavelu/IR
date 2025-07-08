import rrcf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division


class RCTreeForest:
    def __init__(self, num_trees, tree_size, window_size):
        try:
            # Check if input parameters are positive integers
            if not all(isinstance(param, int) and param > 0 for param in [num_trees, tree_size, window_size]):
                raise ValueError("All parameters must be positive integers.")

            # Initialize the Random Cut Tree (RCT) forest with a specified number of trees and tree size
            self.num_trees = num_trees
            self.tree_size = tree_size
            self.shingle_size = window_size

            # Create a list of RCTree instances to form the forest
            self.forest = [rrcf.RCTree() for _ in range(num_trees)]

        except ValueError as e:
            print(f"Error in initialization: {e}")

    def anomaly_detector(self, index, point):
        try:
            # Check if index is a positive integer
            if not isinstance(index, int) or index < 0:
                raise ValueError("Index must be a non-negative integer.")

            # Check if point is a list of numeric values with the correct length
            if not isinstance(point, list) or len(point) != self.shingle_size or not all(isinstance(p, (int, float)) for p in point):
                raise ValueError(f"Point must be a list of {self.shingle_size} numeric values.")

            # Initialize average codisplacement to zero
            avg_codisplacement = 0

            for tree in self.forest:
                # If the tree size exceeds the specified limit, forget the oldest point (FIFO)
                if len(tree.leaves) > self.tree_size:
                    tree.forget_point(index - self.tree_size)

                # Insert the new point into the tree
                tree.insert_point(point, index=index)

                # Compute the codisplacement for the new point
                new_codisplacement = tree.codisp(index)

                # Accumulate the codisplacement across all trees
                avg_codisplacement += new_codisplacement / self.num_trees

            # Return the average codisplacement for the given point
       
            return avg_codisplacement

        except ValueError as e:
            print(f"Error in anomaly detection: {e}")
# Define the number of trees in the Random Cut Tree (RCT) forest
num_trees = 100

# Define the size limit for each tree in the RCT forest
tree_size = 256

## Define the size of the window
shingle_size=10

# Create an instance of the RCTreeForest class with the specified number of trees and tree size
forest = RCTreeForest(num_trees, tree_size,shingle_size)
data_stream =pd.read_csv("cpu.csv",encoding='latin-1')
data_stream =np.array(data_stream)

# Initialize empty lists to store anomaly scores and the current data window
anomaly_score = []
current_window = []
prev_idx=0
first=True

# Iterate through the data stream
for i in range(len(data_stream)):
    # If the index is within the shingle size, populate the initial window with data_stream values
    if i < forest.shingle_size:
        current_window.append(float(data_stream[i]))
        # Initialize anomaly score to 0 for the initial window
        anomaly_score.append(0)
        continue
    else:
        # Update the current window by adding the latest data_stream value and removing the oldest
        current_window.append(float(data_stream[i]))
        current_window = current_window[1:]
    
    # Calculate anomaly score using the RCT forest for the current window
    score = forest.anomaly_detector(i, current_window)
    
    # Print the index for tracking progress (optional)
    #print(i, end=' ')
    
    # Append the calculated anomaly score to the list
    anomaly_score.append(score)
 
    #If there is a sudden peak we can say it is a anomaly
    if i>forest.shingle_size+1 and (score>=1.7*anomaly_score[i-1] or score<=-1.7*anomaly_score[i-1] ): 
        print("Anomaly_Detected_at_index: ", i)


plt.figure(figsize=(10, 6))  # Adjust the width and height as needed

# Plot the original data stream using Matplotlib
plt.plot(np.arange(757), data_stream, label='Data Stream')

# Plot the calculated anomaly scores using Matplotlib
plt.plot(np.arange(757), anomaly_score, label='Anomaly Score', color='red')

# Display the legend to distinguish between the two lines
plt.legend(loc='upper right')


# Display the plot
plt.show()