import pandas as pd
import numpy as np

df = pd.read_csv('breast-cancer.csv')

def standardize(X):
    return (X - np.mean(X)) / np.std(X)

X = df.drop('diagnosis', axis=1)
X = standardize(X)
Y = df['diagnosis']

class node():
  def __init__(self, feature, threshold, gain, value, left, right, ):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.gain = gain
    self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def split_data(self, X, y, feature, threshold):
        left_indices = X.index[X[feature] <= threshold]
        right_indices = X.index[X[feature] > threshold]
        return left_indices, right_indices

    def entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        if np.all(probabilities == 0):
            return 0
        return -np.sum(probabilities * np.log2(probabilities))  # Adding a small constant to avoid log(0)

    def information_gain(self, X, y, feature, threshold):
        left_indices, right_indices = self.split_data(X, y, feature, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        p_left = len(left_indices) / len(X)
        p_right = len(right_indices) / len(X)
        return self.entropy(y) - (p_left * self.entropy(y[left_indices]) + p_right * self.entropy(y[right_indices]))

    def find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = 0
        for feature in X.columns:
            thresholds = np.unique(X[feature])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
      if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
        return node(value=np.unique(y)[0], feature=None, threshold=None, gain=0, left=None, right=None)  # Create a leaf node

      feature, threshold = self.find_best_split(X, y)
      if feature is None:
        return node(value=np.unique(y)[0], feature=None, threshold=None, gain=0, left=None, right=None)  # Create a leaf node if no valid split is found

      left_indices, right_indices = self.split_data(X, y, feature, threshold)
      left_node = self.build_tree(X.loc[left_indices], y[left_indices], depth + 1)
      right_node = self.build_tree(X.loc[right_indices], y[right_indices], depth + 1)
      return node(feature=feature, threshold=threshold, gain=0, value=None, left=left_node, right=right_node)



    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        return [self.make_prediction(x, self.root) for _, x in X.iterrows()]

    def make_prediction(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)
          
def train_test_split(X, y, random_state=41, test_size=0.2):
    n_samples = X.shape[0]

    # Set the seed for the random number generator
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # Determine the size of the test set
    test_size = int(n_samples * test_size)

    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split the features and target arrays into test and train
    # Use .iloc to select rows by their integer positions
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]  
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]  

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    y_true = y_true.to_numpy()
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=41, test_size=0.2)
#create model instance
model = DecisionTree(2, 2)

# Fit the decision tree model to the training data.
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data.
predictions = model.predict(X_test)

# Calculate evaluating metrics
print(f"Model's Accuracy: {accuracy(y_test, predictions)}")
