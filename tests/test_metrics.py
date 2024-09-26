import unittest
from utils.metrics import calculate_metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_metrics_basic(self):
        y_true = [1, 0, 1, 1, 0, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics['precision'], 0.75)
        self.assertAlmostEqual(metrics['recall'], 0.6)
        self.assertAlmostEqual(metrics['f1'], 0.666666, places=5)
        self.assertAlmostEqual(metrics['auc'], 0.75)
        self.assertEqual(metrics['confusion_matrix']['TP'], 3)
        self.assertEqual(metrics['confusion_matrix']['TN'], 3)
        self.assertEqual(metrics['confusion_matrix']['FP'], 1)
        self.assertEqual(metrics['confusion_matrix']['FN'], 1)

    def test_calculate_metrics_single_class(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
        self.assertIsNone(metrics['auc'])  # AUC cannot be computed
        self.assertEqual(metrics['confusion_matrix']['TP'], 4)
        self.assertEqual(metrics['confusion_matrix']['TN'], 0)
        self.assertEqual(metrics['confusion_matrix']['FP'], 0)
        self.assertEqual(metrics['confusion_matrix']['FN'], 0)


if __name__ == '__main__':
    unittest.main()
