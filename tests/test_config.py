import unittest
from utils.config import config


class TestConfig(unittest.TestCase):
    def test_config_load(self):
        self.assertIsNotNone(config)
        self.assertIsInstance(config.config, dict)

    def test_data_config(self):
        data_conf = config.get('data')
        self.assertIn('dataset', data_conf)
        self.assertEqual(data_conf.get('dataset'), 'CIFAR10')
        self.assertIn('batch_size', data_conf)
        self.assertIsInstance(data_conf.get('batch_size'), int)

    def test_attacks_config(self):
        attacks = ['finetune', 'pruning', 'pruning_finetune', 'model_extraction', 'adversarial_training']
        for attack in attacks:
            self.assertIn(attack, config.config['attacks'])
            self.assertIsInstance(config.get('attacks', attack), dict)
            # Additional checks can be added here for specific parameters

    def test_verification_config(self):
        threshold = config.get('verification', 'threshold')
        self.assertIsInstance(threshold, (int, float))
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 100)

    def test_fingerprint_config(self):
        fingerprint_conf = config.get('fingerprint')
        self.assertIn('num_samples', fingerprint_conf)
        self.assertIsInstance(fingerprint_conf.get('num_samples'), int)
        self.assertIn('target_class', fingerprint_conf)
        self.assertIsInstance(fingerprint_conf.get('target_class'), int)

    def test_paths_config(self):
        paths_conf = config.get('paths')
        self.assertIn('attacked_models_dir', paths_conf)
        self.assertIn('fingerprint_file', paths_conf)
        self.assertIsInstance(paths_conf.get('attacked_models_dir'), str)
        self.assertIsInstance(paths_conf.get('fingerprint_file'), str)


if __name__ == '__main__':
    unittest.main()
