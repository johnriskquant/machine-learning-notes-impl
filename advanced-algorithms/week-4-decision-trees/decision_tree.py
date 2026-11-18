"""Entropy-based decision tree classifier with an information-gain split rule.

For ``m`` examples and ``n`` numeric features, training data ``X`` has shape
``(m, n)`` and integer class labels ``y`` have shape ``(m,)``.

Training: ``fit(X, y)``
=======================

    fit(X, y)
      |
      v
    _grow_tree(X, y, depth)
      |
      +-- Stop before splitting?
      |     depth limit reached, too few examples, or labels are all equal
      |       YES -> make a majority-class leaf
      |       NO  -> search for a split
      |
      +-- _best_split(X, y)
      |     try each feature and each observed value as a threshold
      |     score each split by information gain:
      |       parent entropy - weighted child entropy
      |
      +-- No useful split?
      |     no feature/threshold, or gain <= min_impurity_decrease
      |       YES -> make a majority-class leaf
      |       NO  -> divide rows at X[:, feature] <= threshold
      |
      +-- Recursively grow left and right subtrees
            left:  values <= threshold
            right: values > threshold

Prediction: ``predict(X)``
==========================

    For each row x, start at the root:
      |
      +-- At a leaf -> return its class value
      |
      +-- At a split -> follow left if x[feature] <= threshold,
                        otherwise follow right
      |
      +-- Repeat until a leaf is reached
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Node:
    """One tree node: either a feature split or a class-predicting leaf.

    Internal nodes store a feature and threshold, with ``left`` and ``right``
    child nodes. Leaf nodes store their predicted class in ``value``.
    """

    def __init__(
        self,
        feature_index: int | None = None,
        threshold: np.generic | int | float | None = None,
        left: Node | None = None,
        right: Node | None = None,
        value: int | None = None,
    ) -> None:
        """Create a split node or a leaf node."""
        self.feature_index: int | None = feature_index
        self.threshold: np.generic | int | float | None = threshold
        self.left: Node | None = left
        self.right: Node | None = right
        self.value: int | None = value

    def is_leaf(self) -> bool:
        """Return whether this node predicts a class rather than splitting."""
        return self.value is not None


class DecisionTree:
    """A small entropy-based decision tree classifier.

    The tree recursively chooses numeric feature splits that reduce class-label
    entropy. Labels in ``y`` must be non-negative integers because NumPy's
    ``bincount`` is used to count classes. For ``m`` examples and ``n``
    features, ``X`` has shape ``(m, n)`` and ``y`` has shape ``(m,)``.
    """

    def __init__(self, max_depth: int | None = None,min_samples_split: int = 2,min_impurity_decrease: float = 0.0,) -> None:
        """Configure stopping rules for tree growth.

        Args:
            max_depth: Maximum split depth; ``None`` means no depth limit.
            min_samples_split: A node needs at least this many examples to
                be considered for splitting.
            min_impurity_decrease: Minimum information gain required to split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Node | None = None

    def _compute_entropy(self, y: np.ndarray) -> float:
        """Compute class-label entropy, a measure of label mixture.

        Entropy is zero when all examples have the same class and is higher
        when classes are more evenly represented. ``y`` has shape ``(m,)``.
        """
        m = len(y)
        if m == 0:  # An empty child contributes no uncertainty.
            return 0
        else:
            pk = np.bincount(y) / m  # Fraction of examples in each class.
            pk = pk[pk > 0]  # Omit absent classes so log2(0) is never evaluated.
            return (-pk * np.log2(pk)).sum()

    def _information_gain(self, y: np.ndarray, y_left: np.ndarray,y_right: np.ndarray) -> float:
        """Measure entropy reduction from dividing one label vector into two.

        ``y`` has shape ``(m,)``; ``y_left`` and ``y_right`` are one-
        dimensional label arrays whose lengths sum to ``m``.
        """
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        else:
            # Weight each child's entropy by its share of the parent examples.
            w_left = len(y_left) / len(y)
            w_right = len(y_right) / len(y)
            parent_entropy = self._compute_entropy(y)
            child_entropy = w_left * self._compute_entropy(y_left) + w_right * self._compute_entropy(y_right)
        return parent_entropy - child_entropy

    def _best_split(self, X: np.ndarray, y: np.ndarray,) -> tuple[int | None, np.generic | int | float | None, float]:
        """Find the feature and threshold with the greatest information gain.

        Each distinct observed feature value is tested as a threshold. Values
        at or below it go left; larger values go right.

        Args:
            X: Feature matrix with shape ``(m, n)``.
            y: Integer class labels with shape ``(m,)``.

        Returns:
            ``(feature_index, threshold, gain)`` for the best split, or
            ``(None, None, -1)`` if there are no features to test.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_mask = X_column <= threshold
                right_mask = ~left_mask
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    # Keep the split that most reduces label uncertainty.
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _leaf_value(self, y: np.ndarray) -> int:
        """Return the most frequent class, used as the leaf's prediction."""
        return int(np.bincount(y).argmax())

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Build a subtree recursively from aligned data arrays.

        ``X`` has shape ``(m, n)`` and ``y`` has shape ``(m,)``. Each
        recursive call receives the rows assigned to one child.
        """
        stop_conditions = (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        )
        if stop_conditions:
            return Node(value=self._leaf_value(y))
        else:
            best_feature, best_threshold, best_gain = self._best_split(X, y)
            if (
                best_feature is None
                or best_threshold is None
                or best_gain <= self.min_impurity_decrease
            ):
                return Node(value=self._leaf_value(y))
            # Partition both features and labels using the same split decision.
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            left_child = self._grow_tree(X[left_mask], y[left_mask], depth+1)
            right_child = self._grow_tree(X[right_mask], y[right_mask], depth+1)
            return Node(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def fit(self, X: np.ndarray | Sequence[Sequence[int | float]], y: np.ndarray | Sequence[int],) -> DecisionTree:
        """Learn the tree from feature rows ``X`` and integer class labels ``y``.

        Args:
            X: Feature matrix with shape ``(m, n)``.
            y: Non-negative integer class labels with shape ``(m,)``, one
                label for each row in ``X``.

        Returns:
            This fitted tree, so calls can be chained.
        """
        X = np.array(X, copy=True)
        y = np.array(y, copy=True)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """Follow one feature vector of shape ``(n,)`` until reaching a leaf."""
        if node.value is not None:
            return node.value
        if node.feature_index is None or node.threshold is None:
            raise ValueError("Tree node has neither a class value nor a valid split")
        if x[node.feature_index] <= node.threshold:
            if node.left is None:
                raise ValueError("Tree split is missing its left child")
            return self._traverse_tree(x, node.left)
        if node.right is None:
            raise ValueError("Tree split is missing its right child")
        return self._traverse_tree(x, node.right)

    def predict(self, X: np.ndarray | Sequence[Sequence[int | float]]) -> np.ndarray:
        """Predict one class label for each row in ``X``.

        Args:
            X: Feature matrix with shape ``(m, n)``, using the same ``n``
                features and column order as the data passed to :meth:`fit`.

        Returns:
            Predicted labels with shape ``(m,)``.
        """
        if self.root is None:
            raise ValueError("You need to fit the model before calling predict")
        X = np.array(X, copy=True)
        preds = np.array([self._traverse_tree(x, self.root) for x in X])
        return preds


if __name__ == "__main__":
    # Tiny example: the tree can separate the two classes with one threshold.
    X_train = np.array([[1, 1, 1],
                                  [0, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 1],
                                  [1, 1, 1],
                                  [1, 1, 0],
                                  [0, 0, 0],
                                  [1, 1, 0],
                                  [0, 1, 0],
                                  [0, 1, 0]])
    y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

    model = DecisionTree(max_depth=2).fit(X_train, y_train)
    print(model.predict(X_train).tolist())