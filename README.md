# Part 1 – Quadratic Discriminant Analysis (QDA)
This README summarises the first part of the final assignment for the “Introduction to Machine Learning and Data Science” course. The goal of this part is to train a Quadratic Discriminant Analysis (QDA) classifier on grayscale images of clothing items and evaluate its performance on both training and test data.

The dataset consists of images of five categories – T‑shirt/top, trouser, coat, sandal and ankle boot – stored as 28×28 matrices with associated labels. Each image is flattened into a 784‑dimensional feature vector for classification. Training and test sets are provided in TrainData.pkl and TestData.pkl respectively.

## Questions & Implementation
### Q1 – Data loading and exploration
Load data: Read TrainData.pkl and TestData.pkl using pickle. Each dictionary contains keys X (a 28×28×N array of images) and Y (labels).

Display examples: Extract the first index of each unique label and display one sample image per class using matplotlib. This gives an intuition about the appearance of each clothing item.

Reshape images: Transpose the 3‑D tensor to shape (N, 28, 28) and then reshape to (N, 784) to create a matrix of samples by features. This representation is required for QDA.

### Q2 – Parameter estimation and classifier construction
Compute maximum likelihood (ML) estimates: Convert the feature matrix to float64. For each class (five in total) compute the class prior as the class’s proportion of the dataset; the mean vector as the average of all images of that class;
and the covariance matrix using the unbiased sample covariance formula.

Build the classifier: Implement a classify function that, for each sample, computes the log‑likelihood under each class assuming a multivariate normal distribution and returns the class with the highest likelihood.
This realises the QDA decision rule.

Visualise class means: Reshape each class‑mean vector back to 28×28 and display it as an image. These mean images provide insight into the average pattern of each class.

### Q3 – Evaluate on the training set
Accuracy: Compute the fraction of correctly classified samples using an accuracy function.

Confusion matrix: Use sklearn.metrics.confusion_matrix and seaborn to plot a confusion matrix showing how often each true class is predicted as each class.

Precision & recall per class: Implement a helper function to count true positives, false positives and false negatives for each label, and compute precision and recall accordingly.

### Q4 – Evaluate on the test set
Repeat the steps from Q3 using the test data. The model trained on the full training set is used to predict labels for the test set. Report test accuracy, confusion matrix and per‑class precision and recall.

## How to Run
Ensure TrainData.pkl and TestData.pkl are present in the working directory.

Open Part1_QDA.ipynb in Jupyter and run each cell sequentially. The notebook loads the data, visualises samples, computes ML parameters, trains the QDA classifier and outputs evaluation metrics.

The notebook uses numpy, pandas, matplotlib, seaborn and sklearn. Install missing packages via pip install numpy pandas matplotlib seaborn scikit-learn.

## Notes
The dataset contains 24 000 training images and 6 000 test images; each image is grayscale and resized to 28×28.

QDA assumes each class has its own covariance matrix; the model can capture different class variances but is computationally expensive due to large covariance matrices.

The resulting classifier provides high accuracy on this five‑class problem and offers interpretable mean images.




# Part 3 – Regression and Route Optimisation
This README summarises the third part of the final assignment. The task is to learn travel‑time models from historical route data and to use those models to find the fastest route between origin and destination pairs. Two transport modes are considered: pedestrians and cars.

## Data
Two CSV files are provided:

recorded times.csv contains recorded trips with columns: Start, End, Type (either Car or Pedestrian), Date and Travel time. Each row records how long it took to travel between two locations at a particular time.

Test.csv contains test queries with columns Origin, Destination and Date. It does not include travel times; instead, your model must predict the travel time and suggest the fastest route.

### Question 7 – Pedestrian travel time prediction
The first task is to build a regression model that predicts the travel time for pedestrians and to use it to find the optimal route for each test query.

## Pre‑processing
Load data: Read recorded times.csv and Test.csv into pandas DataFrames.

Build adjacency lists: Create dictionaries mapping each start location to the set of reachable destinations separately for pedestrians and cars. These dictionaries form directed graphs for route planning.

Convert dates: Define a helper function convert_to_unix that parses date strings and converts them to Unix timestamps. The training set uses day/month/year format, while the test set uses month/day/year, so the function handles both modes.

Encode categorical variables: Map each distinct location and transport type to a numerical index. Replace Start, End and Type with their numeric codes. Convert dates to Unix time for both training and test sets. Split the data into pedestrian samples (Type = 1) and car samples (Type = 0).

## Polynomial regression model
Feature matrix: Build a helper create_X_matrix that constructs a design matrix for polynomial regression. Given a raw feature matrix X (with columns: time, start index, end index) and a degree d, it returns a matrix with columns \([1, x, x^2, \dots, x^d]) for each input vector.

Train weights: Define calc_w to compute the regression coefficients using the normal equation 

Prediction functions: approx_t predicts travel times for multiple samples, while approx_t_single predicts the time for a single trip using the learned coefficients.

Model fitting: For pedestrians, create a polynomial design matrix of degree 2 and compute the weights w from the pedestrian subset of the training data.

## Route search
To answer test queries, implement a depth‑first search (DFS) that explores all possible routes between the origin and destination within the pedestrian graph. At each step, approximate the time of travelling from one node to the next using the regression model and accumulate the total time. Keep track of the minimum travel time (min_t) and the corresponding path. Skip routes that exceed the current best time. For each test row, call find_best_route and print the optimal route and its estimated travel time.

### Question 8 – Car travel time prediction
The second task repeats the procedure for car trips:

Retrain the regressor using the car subset of the training data (Type = 0) and the same polynomial degree 2 to obtain new weights.

Find best routes using the car_trips graph and the retrained model for each test query. Output the fastest route and travel time for each origin–destination pair.

## How to Run
Place recorded times.csv and Test.csv in the same directory as Part3_RL.ipynb.

Open the notebook in Jupyter and run the cells sequentially. The code will load the data, train regression models for pedestrians and cars, and print the fastest routes and approximate travel times for each test query.

Required packages include pandas, numpy, matplotlib, re (regular expressions) and scikit‑learn (for data splitting if needed). Install missing packages via pip install pandas numpy matplotlib scikit-learn.

## Notes
Dates in the training and test sets use different formats; ensure you pass the correct mode argument (train or test) when converting to Unix time.

Categorical variables (locations and transport type) are encoded as integers to make the regression possible.

The route search uses DFS and may not scale to very large graphs. In practice, one might employ Dijkstra’s algorithm with edge weights given by predicted travel times.

Question 9 in the assignment (not covered in this notebook) likely involves combining transport modes. Extending the solution would require a model trained on both car and pedestrian data and a modified route search.
