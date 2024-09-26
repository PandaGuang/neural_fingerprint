import unittest
from utils.config import config


class TestConfig(unittest.TestCase):
    def test_config_load(self):
        self.assertIsNotNone(config)
        self.assertIsInstance(config.config, dict)

    def test_data_config(self):
        data_conf = config.get('data')
        self.assertIn('dataset', data_conf)
        self.assertIn('batch_size', data_conf)
        self.assertIn('num_workers', data_conf)
        self.assertIn('transforms', data_conf)

    def test_attacks_config(self):
        attacks = ['finetune', 'pruning', 'pruning_finetune', 'model_extraction', 'adversarial_training']
        for attack in attacks:
            self.assertIn(attack, config.config['attacks'])
            self.assertIsInstance(config.get('attacks', attack), dict)

    def test_fingerprint_config(self):
        fingerprint_conf = config.get('fingerprint')
        self.assertIn('num_samples', fingerprint_conf)
        self.assertIn('save_dir', fingerprint_conf)
        self.assertIn('target_class', fingerprint_conf)
        self.assertIn('generation_method', fingerprint_conf)

    def test_verification_config(self):
        verification_conf = config.get('verification')
        self.assertIn('threshold', verification_conf)
        self.assertIsInstance(verification_conf.get('threshold'), (int, float))

    def test_paths_config(self):
        paths_conf = config.get('paths')
        self.assertIn('attacked_models_dir', paths_conf)
        self.assertIn('fingerprint_file', paths_conf)


if __name__ == '__main__':
    unittest.main()
