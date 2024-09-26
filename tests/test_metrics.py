import unittest
from utils.metrics import calculate_metrics

class TestMetrics(unittest.TestCase):
    def test_calculate_metrics_binary(self):
        y_true = [1, 0, 1, 1, 0, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics['precision'], 0.75)
        self.assertAlmostEqual(metrics['recall'], 0.6)
        self.assertAlmostEqual(metrics['f1'], 0.666666, places=5)
        # Since it's binary, AUC and ROC are applicable
        self.assertTrue(0 <= metrics['auc'] <= 1)
        self.assertIn('TP', metrics['confusion_matrix'])
        self.assertIn('TN', metrics['confusion_matrix'])
        self.assertIn('FP', metrics['confusion_matrix'])
        self.assertIn('FN', metrics['confusion_matrix'])

    def test_calculate_metrics_multiclass(self):
        y_true = [0, 1, 2, 1, 0, 2, 1, 0]
        y_pred = [0, 2, 1, 1, 0, 2, 0, 0]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertIsInstance(metrics['precision'], float)
        self.assertIsInstance(metrics['recall'], float)
        self.assertIsInstance(metrics['f1'], float)
        # AUC might not be applicable directly for multiclass without proper handling
        # So, depending on implementation, this might need adjustment
        # For simplicity, we can skip AUC in this test or adjust calculate_metrics accordingly

if __name__ == '__main__':
    unittest.main()
