import argparse
from pathlib import Path

import pandas as pd

# Import metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# define entry type for type hints
type EvalEntry = dict[str, list[str]]


class Scorer:
    def __init__(self):
        print("setting up scorers...")
        self.metrics = [
            (Bleu(4), ["Bleu1", "Bleu2", "Bleu3", "Bleu4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

    def compute_scores_single(self, targets: EvalEntry, results: EvalEntry) -> dict[str, float]:
        if len(targets) != 1 and len(results) != 1:
            raise ValueError("Expected single entry for targets and results")

        total_scores = {}
        for scorer, method in self.metrics:
            score, _ = scorer.compute_score(targets, results)

            if isinstance(method, list):
                for i, m in enumerate(method):
                    total_scores[m] = score[i]
            else:
                total_scores[method] = score

        return total_scores

    def compute_scores_batch(
        self, targets: EvalEntry, results: EvalEntry
    ) -> dict[str, list[float]]:
        total_scores = {}
        for scorer, method in self.metrics:
            _, scores = scorer.compute_score(targets, results)

            if isinstance(method, list):
                for i, m in enumerate(method):
                    total_scores[m] = scores[i]
            else:
                total_scores[method] = scores

        return total_scores


def load_all_data(
    ids: list[int], results_dir: Path, targets_dir: Path
) -> tuple[EvalEntry, EvalEntry]:
    targets = {}
    results = {}

    for id in ids:
        try:
            with open(results_dir / f"{id}.txt", encoding="utf-8", errors="replace") as f:
                pred_caption = f.read().strip()

            with open(targets_dir / f"{id}.txt", encoding="utf-8", errors="replace") as f:
                gt_captions = f.read().strip()

            results[id] = [pred_caption]
            targets[id] = [gt_captions]

        except FileNotFoundError:
            print(f"Warning: Missing file for ID {id}. Skipping.")
            continue
        except UnicodeDecodeError:
            print(f"Warning: Could not decode file for ID {id}. Skipping.")
            continue

    return targets, results


if __name__ == "__main__":
    # Set base paths
    BASE_DIR = Path(__file__).parent.parent.resolve()
    TARGETS_DIR = BASE_DIR / "data/CaptioningDataset"

    # Get args
    parser = argparse.ArgumentParser(description="Captioning script")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["ben", "s1", "s2"],
        required=True,
        help="Dataset to use: ben, s1, or s2",
    )
    parser.add_argument(
        "--results-path",
        "-r",
        type=str,
        default=None,
        help="Path to results directory used for evaluation",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=BASE_DIR / "outputs/metrics",
        help="Path to save output metrics CSV (overrides default)",
    )

    # Parse arguments
    args = parser.parse_args()

    results_dir = (
        Path(args.results_path) if args.results_path is not None else BASE_DIR / f"outputs/{args.mode}"
    )
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ids = sorted(
        [int(f.stem) for f in Path(results_dir).iterdir() if f.is_file() and f.suffix == ".txt"]
    )
    print(f"Found {len(ids)} result files for evaluation in {results_dir}.")

    data = load_all_data(ids, results_dir, TARGETS_DIR)
    targets, results = load_all_data(ids, results_dir, TARGETS_DIR)
    print("Len targets:", len(targets))
    print("Len results:", len(results))

    # Compute scores
    scorer = Scorer()
    scores = scorer.compute_scores_batch(targets, results)
    scores["ids"] = list(targets.keys())  # type: ignore[list]

    # Convert to DF and compute average scores
    df = pd.DataFrame(scores)
    df.set_index("ids", inplace=True)
    mean_scores = df.mean()
    df.loc["mean"] = mean_scores

    # Save scores to CSV
    output_file = output_dir / f"{args.mode}_metrics.csv"
    df.to_csv(output_file, float_format="%.4f", index_label="ids")
    print(f"Saved metrics to {output_file}")
