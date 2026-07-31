import ast
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "src" / "diatom_cascade" / "models.py"
CHECKPOINTS_PATH = PROJECT_ROOT / "src" / "diatom_cascade" / "checkpoints.py"
MODEL_NAMES = {
    "FlatClassifier",
    "ClassToOrderModel",
    "ThreeLevelHierarchicalModel",
    "HCOFGModel",
    "HCOFGSModel",
}

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TIMM_AVAILABLE = importlib.util.find_spec("timm") is not None


def top_level_classes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


class ArchitectureSourceTests(unittest.TestCase):
    def test_models_are_defined_only_in_the_canonical_module(self):
        canonical = top_level_classes(MODELS_PATH)
        self.assertTrue(MODEL_NAMES <= canonical.keys())

        duplicates = []
        for path in (PROJECT_ROOT / "scripts").rglob("*.py"):
            found = MODEL_NAMES & top_level_classes(path).keys()
            duplicates.extend(f"{path.relative_to(PROJECT_ROOT)}:{name}" for name in found)
        self.assertEqual(duplicates, [])

    def test_every_public_constructor_requires_architecture_inputs(self):
        classes = top_level_classes(MODELS_PATH)
        for class_name in MODEL_NAMES:
            constructor = next(
                node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            )
            argument_names = [argument.arg for argument in constructor.args.args]
            with self.subTest(class_name=class_name):
                self.assertEqual(argument_names[-2:], ["model_name", "pretrained"])
                self.assertEqual(constructor.args.defaults, [])

    def test_backbone_schema_has_no_stage_specific_alias(self):
        python_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (PROJECT_ROOT / "src").rglob("*.py")
        )
        self.assertNotIn("class_backbone", python_source)
        self.assertIn('BACKBONE_PREFIX = "backbone."', CHECKPOINTS_PATH.read_text())

    def test_checkpoint_prefix_is_not_caller_configurable(self):
        tree = ast.parse(CHECKPOINTS_PATH.read_text(encoding="utf-8"))
        loader = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_backbone_from_checkpoint"
        )
        self.assertEqual(
            [argument.arg for argument in loader.args.args],
            ["checkpoint_path", "target_module", "map_location"],
        )

    def test_every_training_checkpoint_uses_the_schema_constant(self):
        for path in (PROJECT_ROOT / "scripts" / "train").glob("train_*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(filename=path.name):
                self.assertIn(
                    "'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION",
                    source,
                )
                self.assertNotIn("'checkpoint_schema_version': 2", source)

    def test_progressive_stages_transfer_only_the_canonical_backbone(self):
        stages = {
            "train_H_CO.py": "best_F_C_model.pth",
            "train_H_COF.py": "best_H_CO_model.pth",
            "train_H_COFG.py": "best_H_COF_model.pth",
            "train_H_COFGS.py": "best_H_COFG_model.pth",
        }
        for filename, previous_checkpoint in stages.items():
            source = (PROJECT_ROOT / "scripts" / "train" / filename).read_text(
                encoding="utf-8"
            )
            with self.subTest(filename=filename):
                self.assertEqual(source.count("load_backbone_from_checkpoint("), 1)
                self.assertIn("model.backbone", source)
                self.assertIn(previous_checkpoint, source)
                self.assertIn("pretrained=False", source)


@unittest.skipUnless(
    TORCH_AVAILABLE and TIMM_AVAILABLE,
    "torch and timm are required for instantiated architecture checks",
)
class ArchitectureTorchTests(unittest.TestCase):
    def test_canonical_module_names_are_stable(self):
        from diatom_cascade.models import (
            ClassToOrderModel,
            FlatClassifier,
            HCOFGModel,
            HCOFGSModel,
            ThreeLevelHierarchicalModel,
        )

        cases = [
            (FlatClassifier, (2,), {"backbone", "classifier"}),
            (
                ClassToOrderModel,
                (2, 3),
                {"backbone", "class_classifier", "order_classifier"},
            ),
            (
                ThreeLevelHierarchicalModel,
                (2, 3, 4),
                {"backbone", "class_classifier", "order_classifier", "family_classifier"},
            ),
            (
                HCOFGModel,
                (2, 3, 4, 5),
                {
                    "backbone",
                    "class_classifier",
                    "order_classifier",
                    "family_classifier",
                    "genus_classifier",
                },
            ),
            (
                HCOFGSModel,
                (2, 3, 4, 5, 6),
                {
                    "backbone",
                    "class_classifier",
                    "order_classifier",
                    "family_classifier",
                    "genus_classifier",
                    "species_classifier",
                },
            ),
        ]
        for model_class, dimensions, expected_modules in cases:
            with self.subTest(model_class=model_class.__name__):
                model = model_class(
                    *dimensions,
                    model_name="efficientnet_b0",
                    pretrained=False,
                )
                self.assertEqual(set(model._modules), expected_modules)


if __name__ == "__main__":
    unittest.main()
