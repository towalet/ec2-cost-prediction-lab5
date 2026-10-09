"""Run final checks with: python -m unittest -v test_lab.py."""

import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from analysis import cost_comparison, instance_family, on_demand_outliers, predict_cost, train_model
from data_preparation import COST_COLUMNS, FEATURE_COLUMNS, load_data


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.data, cls.usable = load_data()
        cls.model, cls.metrics, cls.results, cls.training_count = train_model(cls.usable)

    def test_dataset_and_cleaning(self):
        self.assertEqual(self.raw.shape, (861, 11))
        self.assertEqual(len(self.usable), 841)
        for column in COST_COLUMNS:
            for original, cleaned in zip(self.raw[column], self.data[column]):
                if original == "unavailable":
                    self.assertTrue(np.isnan(cleaned))
                else:
                    self.assertEqual(cleaned, float(original.split()[0][1:].replace(",", "")))
        self.assertFalse(self.usable[FEATURE_COLUMNS + ["On Demand"]].isna().any().any())

    def test_analysis_and_exact_families(self):
        for family in ["T2", "T3"]:
            selected = instance_family(self.data, family)
            self.assertEqual(len(selected), 7)
            self.assertTrue(selected["API Name"].str.startswith(family.lower() + ".").all())
        outliers, lower, upper = on_demand_outliers(self.data)
        expected = self.data["On Demand"].map(lambda cost: cost < lower or cost > upper)
        self.assertEqual(outliers.index.tolist(), self.data.index[expected].tolist())
        comparison = cost_comparison(self.data)
        self.assertTrue(comparison["On Demand"].is_monotonic_increasing)
        row = comparison.iloc[0]
        self.assertAlmostEqual(row["Savings (%)"], (1 - row["Linux Reserved cost"] / row["On Demand"]) * 100)

    def test_held_out_metrics_and_regression(self):
        self.assertEqual((self.training_count, len(self.results)), (672, 169))
        errors = self.results["Actual"] - self.results["Predicted"]
        self.assertAlmostEqual(self.metrics["MAE"], np.abs(errors).mean())
        self.assertAlmostEqual(self.metrics["MSE"], (errors ** 2).mean())
        self.assertAlmostEqual(self.metrics["RMSE"], np.sqrt((errors ** 2).mean()))
        # Independent least-squares solution on the rows not held out.
        training = self.usable.drop(index=self.results.index)
        design = np.column_stack([np.ones(len(training)), training[FEATURE_COLUMNS].to_numpy()])
        coefficients = np.linalg.lstsq(design, training["On Demand"].to_numpy(), rcond=None)[0]
        np.testing.assert_allclose([self.model.intercept_, *self.model.coef_], coefficients, rtol=1e-9)
        self.assertAlmostEqual(predict_cost(self.model, 4, 2), -0.399339484289847, places=6)

    def test_dashboard_inputs_families_and_boundaries(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "$-0.3993")
        self.assertEqual(len(app.warning), 1)
        app.number_input(key="memory").set_value(64.0)
        app.number_input(key="vcpus").set_value(16).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, f"${predict_cost(self.model, 64, 16):.4f}")
        self.assertEqual(len(app.warning), 0)
        app.selectbox(key="family").select("T3").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("7 T3 instances" in element.value for element in app.markdown))
        for memory, cpus in [(0.5, 1), (float(self.usable['Instance Memory'].max()), int(self.usable['vCPUs'].max()))]:
            app.number_input(key="memory").set_value(memory)
            app.number_input(key="vcpus").set_value(cpus).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[0].value, f"${predict_cost(self.model, memory, cpus):.4f}")


if __name__ == "__main__":
    unittest.main()
