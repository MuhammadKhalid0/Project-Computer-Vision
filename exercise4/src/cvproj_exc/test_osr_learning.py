import unittest

import numpy as np

from cvproj_exc.osr_learning import UNKNOWN_LABEL, mpl_training, spl_training


class TestOSRLearning(unittest.TestCase):
    def setUp(self) -> None:
        n_samples = 100
        n_classes = 5
        n_features = 32
        self.n_test_samples = 50
        self.dummy_train_x = np.random.rand(n_samples, n_features)
        self.dummy_train_y = np.random.randint(UNKNOWN_LABEL, n_classes, size=(n_samples,))
        self.dummy_test_x = np.random.rand(self.n_test_samples, n_features)
        self.dummy_osr_train_x = np.random.rand(n_samples, n_features)
        third = n_samples // 3
        self.dummy_osr_train_x[:third] *= 10.0
        self.dummy_osr_train_x[third : third * 2] *= -10.0
        self.dummy_osr_train_x[third * 2 :] *= 100.0
        self.dummy_osr_train_y = np.zeros((n_samples,), dtype=int)
        self.dummy_osr_train_y[third : third * 2] = 1
        self.dummy_osr_train_y[third * 2 :] = UNKNOWN_LABEL
        self.dummy_osr_test_x = np.random.rand(n_samples, n_features)
        self.dummy_osr_test_x[:third] *= 10.0
        self.dummy_osr_test_x[third : third * 2] *= -10.0
        self.dummy_osr_test_x[third * 2 :] *= 100.0

    def test_spl_training_returns_callable(self):
        self.assertTrue(
            callable(spl_training(self.dummy_train_x, self.dummy_train_y)),
            msg="You modified the interface, spl_training has to return a callable!",
        )

    def test_spl_predict_output(self):
        spl_predict_fn = spl_training(self.dummy_train_x, self.dummy_train_y)
        spl_predict_out = spl_predict_fn(self.dummy_test_x)
        self.__test_predict_fn(spl_predict_out, method="spl")

    def test_spl_predict_unknown(self):
        spl_predict_fn = spl_training(self.dummy_train_x, self.dummy_train_y)
        spl_predict_out = spl_predict_fn(self.dummy_osr_test_x)[0]
        self.assertTrue(
            np.any(spl_predict_out == UNKNOWN_LABEL),
            msg="You did not predict any unknown label. Depending on your classifier, "
            "this must not be an error. However, make sure that your unknowns are predicted "
            f"with label '{UNKNOWN_LABEL}'.",
        )

    def test_mpl_training_returns_callable(self):
        self.assertTrue(
            callable(mpl_training(self.dummy_train_x, self.dummy_train_y)),
            msg="You modified the interface, mpl_training has to return a callable!",
        )

    def test_mpl_predict_output(self):
        mpl_predict_fn = mpl_training(self.dummy_train_x, self.dummy_train_y)
        mpl_predict_out = mpl_predict_fn(self.dummy_test_x)
        self.__test_predict_fn(mpl_predict_out, method="mpl")

    def test_mpl_predict_unknown(self):
        mpl_predict_fn = mpl_training(self.dummy_train_x, self.dummy_train_y)
        mpl_predict_out = mpl_predict_fn(self.dummy_osr_test_x)[0]
        self.assertTrue(
            np.any(mpl_predict_out == UNKNOWN_LABEL),
            msg="You did not predict any unknown label. Depending on your classifier, "
            "this must not be an error. However, make sure that your unknowns are predicted "
            f"with label '{UNKNOWN_LABEL}'.",
        )

    def __test_predict_fn(self, predict_out, method):
        # spl_predict_fn and mpl_predict_fn should output y_pred and y_score
        self.assertEqual(
            len(predict_out),
            2,
            msg="'{}_predict_fn' should return 2 objects, but it returned '{}' objects".format(
                method, len(predict_out)
            ),
        )

        # output should be numpy arrays
        y_pred, y_score = predict_out
        self.assertTrue(
            isinstance(y_pred, np.ndarray),
            msg="'{}_predict_fn' should return y_pred as a numpy array, "
            "but it returned a '{}'".format(method, type(y_pred)),
        )
        self.assertTrue(
            isinstance(y_score, np.ndarray),
            msg="'{}_predict_fn' should return y_score as a numpy array, "
            "but it returned a '{}'".format(method, type(y_score)),
        )

        # both arrays should be 1-dimensional
        self.assertEqual(
            y_pred.ndim,
            1,
            msg="'{}_predict_fn' should return y_pred as a 1d array, "
            "but it returned a {}d array".format(method, y_pred.ndim),
        )
        self.assertEqual(
            y_score.ndim,
            1,
            msg="'{}_predict_fn' should return y_sore as a 1d array, "
            "but it returned a {}d array".format(method, y_score.ndim),
        )

        # both arrays should have the same length as the number of test samples
        self.assertEqual(
            len(y_pred),
            self.n_test_samples,
            msg="'{}_predict_fn' should return y_pred with the same length as the test samples, "
            "but it returned '{}' instead of '{}'".format(method, len(y_pred), self.n_test_samples),
        )
        self.assertEqual(
            len(y_score),
            self.n_test_samples,
            msg="'{}_predict_fn' should return y_score with the same length as the test samples, "
            "but it returned '{}' instead of '{}'".format(
                method, len(y_score), self.n_test_samples
            ),
        )
