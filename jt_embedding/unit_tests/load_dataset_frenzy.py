import unittest
from ai_for_math.data.dataset import load_dataset
from transformers import AutoTokenizer

class TestFrenzyMathDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        self.dataset = load_dataset(
            dataset_name="frenzy",
            fl_tokenizer=self.tokenizer,
            nl_tokenizer=self.tokenizer
        )

    def test_dataset_loaded(self):
        self.assertIsNotNone(self.dataset, "Dataset failed to load")

    def test_formal_theorems_accessible(self):
        self.assertTrue(hasattr(self.dataset, "formal_theorems"))
        self.assertIsInstance(self.dataset.formal_theorems, list)

    def test_informal_theorems_accessible(self):
        self.assertTrue(hasattr(self.dataset, "informal_theorems"))
        self.assertIsInstance(self.dataset.informal_theorems, list)

    def test_theorems_not_empty(self):
        self.assertGreater(len(self.dataset.formal_theorems), 0, "Formal theorems list is empty")
        self.assertGreater(len(self.dataset.informal_theorems), 0, "Informal theorems list is empty")

if __name__ == "__main__":
    unittest.main()
