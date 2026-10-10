import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banking77 import label_names, read_csv


class Banking77Tests(unittest.TestCase):
    def test_preserves_label_order_and_csv_content(self):
        names = ['z_label', 'a_label'] + [f'label_{i}' for i in range(75)]
        source = f"features = datasets.ClassLabel(names={names!r})"
        self.assertEqual(label_names(source), names)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'category'])
                writer.writerow(['Text, with comma\nand newline', 'a_label'])
                writer.writerow(['Second', 'z_label'])
            dataset = read_csv(path, names)
            self.assertEqual(dataset['label'], [1, 0])
            self.assertEqual(dataset[0]['text'], 'Text, with comma\nand newline')
            self.assertEqual(dataset.features['label'].names, names)

    def test_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.csv'
            path.write_text('text,category\nhello,unknown\n')
            with self.assertRaises(KeyError):
                read_csv(path, ['known'])
