import numpy as np

class Oracle:

    def __init__(self, p, threshold=1e-5):
        self.p = p
        self.threshold = threshold

    def check_shapes(self, a, b, c):
        shapes = [x.shape for x in [a, b, c]]
        if not all([len(shape) == len(shapes[0]) for shape in shapes]): 
            raise RuntimeError('Number of dimensions mismatch:', a.shape, b.shape, c.shape)
        shape_a = a.shape
        if not all([s == shape_a for s in shapes]):
            raise RuntimeError('Shape mismatch:', a.shape, b.shape, c.shape)

    # Return 0 if preds_a is better and 1 if preds_b is better
    def get_labels(self, y_true, preds_a, preds_b):
        preds_a = preds_a.squeeze()
        preds_b = preds_b.squeeze()
        y_true = y_true.squeeze()

        self.check_shapes(preds_a, preds_b, y_true)

        errors_a = (preds_a - y_true)**2
        errors_b = (preds_b - y_true)**2

        # Positive if b is better than a
        errors = (errors_a-errors_b)

        # How many to take at least
        B0 = int(self.p * len(errors))
        # How many to take at max
        Bmax = (errors > -self.threshold).sum()

        B = max(Bmax, B0)

        label = np.zeros((len(errors)))
        label[np.argsort(-errors)[:B]] = 1

        return label.astype(np.int8)
        

