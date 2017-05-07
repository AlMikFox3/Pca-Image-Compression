import numpy as np
class Normalization:

    def __init__(self, train_x, train_y=None):

        if train_y is not None:
            self.y_mean = np.mean(train_y)
            self.y_std = np.std(train_y)
        else:
            self.y_mean = None
            self.y_std = None
        self.x_means = list(np.mean(train_x, 0))
        self.x_stds = list(np.std(train_x, 0))

        self.norm_train_x = self.normalize_dataset_x(train_x)
        if train_y is not None:
            self.norm_train_y = self.normalize_dataset_y(train_y)
        else:
            self.norm_train_y = None

    def normalized_dataset(self):
        
        if self.norm_train_y is None:
            return self.norm_train_x
        return self.norm_train_x, self.norm_train_y

    def normalize_x(self, x):
       
        return [(float(x[i]) - self.x_means[i]) / self.x_stds[i] for i in range(len(x))]

    def normalize_dataset_x(self, data_x):
        norm_dataset_x = np.array([self.normalize_x(x) for x in data_x])
        return norm_dataset_x

    def denormalize_x(self, x):
        return [float(x[i]) * self.x_stds[i] + self.x_means[i] for i in range(len(x))]

    def denormalize_dataset_x(self, norm_data_x):
        denorm_dataset_x = np.array([self.denormalize_x(x) for x in norm_data_x])
        return denorm_dataset_x

    def normalize_y(self, y):
        if self.y_mean is None or self.y_std is None:
            return None
        return (y - self.y_mean) / self.y_std

    def normalize_dataset_y(self, data_y):
        norm_dataset_y = [self.normalize_y(y) for y in data_y]
        return norm_dataset_y

    def denormalize_y(self, y):
        
        if self.y_mean is None or self.y_std is None:
            return None
        return y * self.y_std + self.y_mean

    def denormalize_dataset_y(self, norm_data_y):
        denorm_dataset_y = [self.denormalize_y(y) for y in norm_data_y]
        return denorm_dataset_y