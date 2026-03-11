import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Dict, Optional


def load_coco_annotations(json_path: Path) -> Dict[str, Any]:
    """
    Load a COCO-format annotation file.

    Args:
        json_path: Path to the input COCO annotation JSON file.

    Returns:
        A dictionary containing COCO-format annotations.
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco_annotations(coco_data: Dict[str, Any], json_path: Path) -> None:
    """
    Save COCO-format annotations to a JSON file.

    Args:
        coco_data: COCO-format annotation dictionary.
        json_path: Output JSON file path.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)


def sample_coco_images(
    coco_data: Dict[str, Any],
    subset_ratio: float,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Sample a subset of images from a COCO-format dataset and keep only the
    corresponding annotations.

    The sampling is performed at the image level. All annotations associated
    with the sampled images are retained.

    Args:
        coco_data: Input COCO-format annotation dictionary.
        subset_ratio: Fraction of images to retain, in the range (0, 1].
        random_seed: Random seed for reproducibility.

    Returns:
        A new COCO-format dictionary containing the sampled subset.

    Raises:
        ValueError: If subset_ratio is not in the range (0, 1].
        KeyError: If required COCO fields are missing.
    """
    if not (0.0 < subset_ratio <= 1.0):
        raise ValueError(f"`subset_ratio` must be in (0, 1], but got {subset_ratio}.")

    if "images" not in coco_data or "annotations" not in coco_data or "categories" not in coco_data:
        raise KeyError("Input COCO data must contain 'images', 'annotations', and 'categories' fields.")

    all_images = coco_data["images"]
    num_total_images = len(all_images)

    if num_total_images == 0:
        raise ValueError("The input dataset contains no images.")

    if subset_ratio == 1.0:
        return deepcopy(coco_data)

    num_subset_images = math.floor(num_total_images * subset_ratio)
    num_subset_images = max(1, num_subset_images)

    rng = Random(random_seed)
    sampled_images = rng.sample(all_images, num_subset_images)
    sampled_image_ids = {img["id"] for img in sampled_images}

    sampled_annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] in sampled_image_ids
    ]

    subset_coco_data = {
        "info": deepcopy(coco_data.get("info", {})),
        "licenses": deepcopy(coco_data.get("licenses", [])),
        "images": deepcopy(sampled_images),
        "annotations": deepcopy(sampled_annotations),
        "categories": deepcopy(coco_data["categories"]),
    }

    return subset_coco_data


def merge_coco_categories(
    coco_data: Dict[str, Any],
    merged_category_name: str = "object",
    merged_category_id: int = 1,
) -> Dict[str, Any]:
    """
    Merge all categories in a COCO-format dataset into a single category.

    Args:
        coco_data: Input COCO-format annotation dictionary.
        merged_category_name: Name of the merged category.
        merged_category_id: Category ID assigned to the merged category.

    Returns:
        A new COCO-format dictionary with all annotations mapped to one category.

    Raises:
        KeyError: If required COCO fields are missing.
    """
    if "annotations" not in coco_data or "categories" not in coco_data:
        raise KeyError("Input COCO data must contain 'annotations' and 'categories' fields.")

    merged_coco_data = deepcopy(coco_data)

    merged_coco_data["categories"] = [
        {
            "id": merged_category_id,
            "name": merged_category_name,
            "supercategory": merged_category_name,
        }
    ]

    for ann in merged_coco_data["annotations"]:
        ann["category_id"] = merged_category_id

    return merged_coco_data


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a COCO-format annotation file by optionally sampling an "
            "image subset and merging all categories into a single class."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input COCO annotation JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the processed COCO annotation JSON file.",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="Fraction of images to retain, in the range (0, 1]. Default: 1.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for image sampling. Default: 42.",
    )
    parser.add_argument(
        "--merge-categories",
        action="store_true",
        help="Merge all categories into a single category.",
    )
    parser.add_argument(
        "--merged-category-name",
        type=str,
        default="object",
        help="Name of the merged category. Default: 'object'.",
    )
    parser.add_argument(
        "--merged-category-id",
        type=int,
        default=1,
        help="ID of the merged category. Default: 1.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for command-line execution.
    """
    args = parse_args()

    coco_data = load_coco_annotations(args.input)

    processed_data = sample_coco_images(
        coco_data=coco_data,
        subset_ratio=args.subset_ratio,
        random_seed=args.seed,
    )

    if args.merge_categories:
        processed_data = merge_coco_categories(
            coco_data=processed_data,
            merged_category_name=args.merged_category_name,
            merged_category_id=args.merged_category_id,
        )

    save_coco_annotations(processed_data, args.output)

    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Subset ratio: {args.subset_ratio}")
    print(f"Random seed:  {args.seed}")
    print(f"Num images:   {len(processed_data.get('images', []))}")
    print(f"Num anns:     {len(processed_data.get('annotations', []))}")
    print(f"Num cats:     {len(processed_data.get('categories', []))}")


if __name__ == "__main__":
    main()