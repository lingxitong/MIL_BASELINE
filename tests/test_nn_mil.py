import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch
import yaml

from modules.NN_MIL.nn_mil import NN_MIL
from utils.nnmil_utils import fixed_bag_collate


class TestNNMIL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = NN_MIL(in_dim=8, hidden_dim=4, num_classes=3, dropout=0.0,
                            eval_stride_divisor=2, cover_seed=11)

    def test_padding_mask_matches_unpadded_evaluation(self):
        features = torch.randn(1, 3, 8)
        padded = torch.cat([features, torch.zeros(1, 2, 8)], dim=1)
        mask = torch.tensor([[True, True, True, False, False]])
        self.model.eval()
        raw_logits = self.model(features)["logits"]
        padded_logits = self.model(padded, valid_mask=mask)["logits"]
        self.assertTrue(torch.allclose(raw_logits, padded_logits, atol=1e-6))

    def test_training_supports_batched_fixed_bags_and_backward(self):
        bags = torch.randn(3, 6, 8)
        mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2, [True] * 5 + [False]])
        self.model.train()
        logits = self.model(bags, valid_mask=mask)["logits"]
        self.assertEqual(tuple(logits.shape), (3, 3))
        logits.sum().backward()
        self.assertIsNotNone(self.model.V.weight.grad)

    def test_evaluation_returns_deterministic_ensemble_uncertainty(self):
        self.model.eval()
        features = torch.randn(2, 5, 8)
        first = self.model(features)
        second = self.model(features)
        self.assertTrue(torch.equal(first["logits"], second["logits"]))
        self.assertGreater(first["chunk_logits"].shape[0], 1)
        self.assertTrue(torch.all(first["mutual_information"] >= 0))

    def test_fixed_bag_collate_subsamples_and_masks_padding(self):
        long_bag = torch.randn(7, 8)
        short_bag = torch.randn(3, 8)
        bags, labels, mask = fixed_bag_collate([(long_bag, torch.tensor(0)), (short_bag, torch.tensor(1))], 5)
        self.assertEqual(tuple(bags.shape), (2, 5, 8))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(mask.sum(dim=1).tolist(), [5, 3])

    def test_train_and_test_entrypoints(self):
        """A one-epoch end-to-end smoke test of the MIL_BASELINE integration."""
        repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as directory:
            rows = {}
            for split, labels in {"train": [0, 1, 0, 1], "val": [0, 1], "test": [0, 1]}.items():
                paths = []
                for index, label in enumerate(labels):
                    path = os.path.join(directory, f"{split}_{index}.pt")
                    torch.save(torch.randn(3 + index, 8), path)
                    paths.append(path)
                rows[f"{split}_slide_path"] = pd.Series(paths)
                rows[f"{split}_label"] = pd.Series(labels)
            dataset_path = os.path.join(directory, "dataset.csv")
            pd.DataFrame(rows).to_csv(dataset_path, index=False)
            logs = os.path.join(directory, "logs")
            config = {
                "General": {"MODEL_NAME": "NN_MIL", "seed": 1, "num_classes": 2, "num_epochs": 1,
                            "device": 0, "num_workers": 0, "best_model_metric": "macro_auc",
                            "earlystop": {"use": False, "patience": 2, "metric": "macro_auc"}},
                "Dataset": {"DATASET_NAME": "toy", "dataset_csv_path": dataset_path, "dataset_root_dir": {}},
                "Logs": {"log_root_dir": logs},
                "Model": {
                    "fixed_bag_size": 4, "auto_bag_size_factor": 0.5, "batch_size": 2,
                    "task_aware_sampler": True, "in_dim": 8, "hidden_dim": 4, "dropout": 0.0,
                    "activation": "softmax", "feature_select": True, "eval_stride_divisor": 2,
                    "cover_shuffle": True, "cover_seed": 3, "criterion": "ce",
                    "optimizer": {"which": "adam", "adam_config": {"lr": 0.001, "weight_decay": 0.0},
                                  "adamw_config": {"lr": 0.001, "weight_decay": 0.0}},
                    "scheduler": {"warmup": 1, "which": "none", "step_config": {"step_size": 2, "gamma": 0.9},
                                  "multi_step_config": {"milestones": [2], "gamma": 0.9},
                                  "exponential_config": {"gamma": 0.9}, "cosine_config": {"T_max": 2, "eta_min": 0.0}},
                },
            }
            config_path = os.path.join(directory, "nn_mil.yaml")
            with open(config_path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream)
            subprocess.run([sys.executable, "train_mil.py", "--yaml_path", config_path], cwd=repository_root, check=True)
            checkpoint = next(os.path.join(root, file) for root, _, files in os.walk(logs) for file in files
                              if file.startswith("Last_EPOCH_"))
            test_log_dir = os.path.join(directory, "test_output")
            subprocess.run([sys.executable, "test_mil.py", "--yaml_path", config_path,
                            "--test_dataset_csv", dataset_path, "--model_weight_path", checkpoint,
                            "--test_log_dir", test_log_dir], cwd=repository_root, check=True)
            prediction_path = os.path.join(test_log_dir, "NN_MIL_predictions.csv")
            self.assertTrue(os.path.isfile(prediction_path))
            predictions = pd.read_csv(prediction_path)
            self.assertIn("mutual_information", predictions.columns)
            self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
