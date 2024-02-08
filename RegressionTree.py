import numpy as np


class Node:

    def __init__(self, parent=None, feature_number=0, treshold=0, is_list=False) -> None:
        self.feature_number = feature_number
        self.treshold = treshold
        self.parent = parent
        self.left_node = None
        self.right_node = None
        self.is_list = is_list


class DecisionTreeRegressor:

    def __init__(self, max_depth=0) -> None:
        self.root = Node()
        self.max_depth = max_depth
        self.first_fit_call = False

    def infrom_gain(self, X_par_sz, split, y_pref_sum, y_sq_pref_sum):
        l_sz = split + 1
        r_sz = X_par_sz - l_sz
        p_dispers = y_sq_pref_sum[-1] / X_par_sz - y_pref_sum[-1] ** 2 / X_par_sz**2
        l_dispers = y_sq_pref_sum[l_sz] / l_sz - y_pref_sum[l_sz] ** 2 / l_sz**2
        r_dispers = (y_sq_pref_sum[-1] - y_sq_pref_sum[l_sz]) / r_sz - (
            y_pref_sum[-1] - y_pref_sum[l_sz]
        ) ** 2 / r_sz**2

        return X_par_sz * p_dispers - l_sz * l_dispers - r_sz * r_dispers

    def find_optimal_treshold(self, X, Y):
        x_y = np.hstack((X, np.reshape(Y, (-1, 1))))
        n = X.shape[0]
        feature_count = X.shape[1]
        opt_feature = 0
        mx_ig = 0
        opt_value = 0
        for feature in range(feature_count):
            x_y = x_y[np.argsort(x_y[:, feature])]

            y_pref_sum = np.zeros(n + 1)
            y_sq_pref_sum = np.zeros(n + 1)

            for j in range(1, n + 1):
                y_pref_sum[j] += y_pref_sum[j - 1] + x_y[j - 1][-1]
                y_sq_pref_sum[j] += y_sq_pref_sum[j - 1] + x_y[j - 1][-1] ** 2

            split = 0
            while split < n - 1:
                while split < n - 2 and X[split][feature] == X[split + 1][feature]:
                    split += 1
                curr_ig = self.infrom_gain(n, split, y_pref_sum, y_sq_pref_sum)

                if curr_ig > mx_ig:
                    mx_ig = curr_ig
                    opt_feature = feature
                    opt_value = (X[split][feature] + X[split + 1][feature]) / 2

                split += 1

        return (opt_feature, opt_value)

    def insertNode(self, parent, treshold, is_left, islist=False):
        vert = Node(
            parent=parent, feature_number=treshold[0], treshold=treshold[1], is_list=islist
        )
        if is_left:
            parent.left_node = vert
        else:
            parent.right_node = vert
        return vert

    def fit(self, X, Y, node, is_left=True, depth=0):
        if not self.first_fit_call:
            self.first_fit_call = True
            piv = self.find_optimal_treshold(X, Y)
            self.root.feature_number, self.root.treshold = piv[0], piv[1]
            n = X.shape[0]
            x_l, y_l = [], []
            x_r, y_r = [], []
            for i in range(n):
                if X[i][piv[0]] < piv[1]:
                    x_l.append(X[i])
                    y_l.append(Y[i])
                else:
                    x_r.append(X[i])
                    y_r.append(Y[i])
            x_l, y_l, x_r, y_r = (
                np.array(x_l),
                np.array(y_l),
                np.array(x_r),
                np.array(y_r),
            )

            self.fit(x_l, y_l, self.root, True, depth + 1)
            self.fit(x_r, y_r, self.root, False, depth + 1)
            return

        if depth == self.max_depth - 1 or X.shape[0] <= 1:
            treshold = (0, np.mean(Y))
            self.insertNode(node, treshold, is_left=is_left, islist=True)
            return
        piv = self.find_optimal_treshold(X, Y)
        v = self.insertNode(node, piv, is_left=is_left)
        n = X.shape[0]
        x_l, y_l = [], []
        x_r, y_r = [], []

        for i in range(n):
            if X[i][piv[0]] < piv[1]:
                x_l.append(X[i])
                y_l.append(Y[i])
            else:
                x_r.append(X[i])
                y_r.append(Y[i])

        x_l, y_l, x_r, y_r = np.array(x_l), np.array(y_l), np.array(x_r), np.array(y_r)
        self.fit(x_l, y_l, v, True, depth + 1)
        self.fit(x_r, y_r, v, False, depth + 1)

    def search_list(self, x, vert):
        if vert.is_list:
            return vert.treshold
        if x[vert.feature_number] <= vert.treshold:
            return self.search_list(x, vert.left_node)
        else:
            return self.search_list(x, vert.right_node)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.search_list(x, self.root))
        predictions = np.array(predictions)
        return predictions


    def printTree(self, vert):
        if not vert.is_list:
            print("-----------------")
            print(f"feature: {vert.feature_number}  treshold: {vert.treshold} ")
            if vert.left_node != None:
                self.printTree(vert.left_node)
            if vert.right_node != None:
                self.printTree(vert.right_node)
        else:
            print("------------------")
            print(f"LIST {vert.treshold}")


# model = DecisionTreeRegressor(max_depth = 10)
