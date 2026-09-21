"""Tests for the visualization outputs."""
import hashlib
import unittest
from collections import Counter

from src.run_visualizations import run_visualizations
from src.visualize_results import (
    FIGURES_DIR,
    PAIRWISE_PATH,
    SUMMARY_PATH,
    load_pairwise,
    load_summary,
)

EXPECTED_STEMS = [
    "01_overall_match_scores",
    "02_match_distribution",
    "03_failure_modes",
    "04_person_comparison",
]


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GenerateFiguresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary_hash_before = _hash(SUMMARY_PATH)
        cls.pairwise_hash_before = _hash(PAIRWISE_PATH)
        run_visualizations()

    def test_all_png_figures_generated_and_non_empty(self):
        for stem in EXPECTED_STEMS:
            path = FIGURES_DIR / f"{stem}.png"
            self.assertTrue(path.exists(), f"Missing {path}")
            self.assertGreater(path.stat().st_size, 0)

    def test_no_svg_files_produced(self):
        self.assertEqual(list(FIGURES_DIR.glob("*.svg")), [])

    def test_figures_directory_contains_exactly_the_four_pngs(self):
        actual = {path.name for path in FIGURES_DIR.iterdir()}
        expected = {f"{stem}.png" for stem in EXPECTED_STEMS}
        self.assertEqual(actual, expected)

    def test_source_files_untouched(self):
        self.assertEqual(_hash(SUMMARY_PATH), self.summary_hash_before)
        self.assertEqual(_hash(PAIRWISE_PATH), self.pairwise_hash_before)


class FrozenValueTests(unittest.TestCase):
    def setUp(self):
        self.summary = load_summary()
        self.pairwise_rows = load_pairwise()

    def test_overall_scores_match_frozen_values(self):
        self.assertEqual(self.summary["automated_fidelity_mean"], 2.6333333333333333)
        self.assertEqual(self.summary["human_fidelity_mean"], 2.533333333333333)

    def test_match_distribution_sums_to_thirty(self):
        self.assertEqual(len(self.pairwise_rows), 30)
        counts = Counter(row["human_fidelity_score"] for row in self.pairwise_rows)
        self.assertEqual(sum(counts.values()), 30)

    def test_match_distribution_values_are_exact(self):
        counts = Counter(row["human_fidelity_score"] for row in self.pairwise_rows)
        self.assertEqual(counts.get(5, 0), 0)
        self.assertEqual(counts.get(4, 0), 7)
        self.assertEqual(counts.get(3, 0), 10)
        self.assertEqual(counts.get(2, 0), 5)
        self.assertEqual(counts.get(1, 0), 8)

    def test_failure_mode_percentages_match_frozen_summary(self):
        evidence = self.summary["failure_evidence"]
        self.assertEqual(evidence["human_unsupported_detail_rate"], 100.0)
        self.assertEqual(evidence["human_omission_rate"], 73.33333333333333)
        self.assertEqual(evidence["human_contradiction_rate"], 36.666666666666664)

    def test_person_comparison_contains_exactly_three_people(self):
        person_ids = {row["person_id"] for row in self.summary["person_summary"]}
        self.assertEqual(person_ids, {"c_human", "g_human", "s_human"})

    def test_person_scores_match_frozen_values(self):
        person_summary = {row["person_id"]: row for row in self.summary["person_summary"]}
        expected = {
            "c_human": (2.90, 2.70, "Low"),
            "g_human": (2.30, 2.50, "Low"),
            "s_human": (2.70, 2.40, "Medium"),
        }
        for person_id, (automated, human, consistency) in expected.items():
            row = person_summary[person_id]
            self.assertAlmostEqual(row["automated_fidelity_mean"], automated)
            self.assertAlmostEqual(row["human_fidelity_mean"], human)
            self.assertEqual(self.summary["person_consistency"][person_id], consistency)


if __name__ == "__main__":
    unittest.main()
