import argparse
import json
import os
import sys


VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_train_indices(train_arg):
    if train_arg is None:
        raise ValueError("Missing --train / -train argument.")

    tokens = [token.strip() for token in train_arg.split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError("Training index list is empty.")

    try:
        train_indices = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Training indices must be comma-separated integers.") from exc

    duplicate_indices = []
    seen = set()
    for idx in train_indices:
        if idx in seen and idx not in duplicate_indices:
            duplicate_indices.append(idx)
        seen.add(idx)

    if duplicate_indices:
        raise ValueError(f"Duplicate training indices found: {duplicate_indices}")

    return train_indices


def count_images(images_path):
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    image_files = [
        file_name
        for file_name in os.listdir(images_path)
        if os.path.splitext(file_name)[1].lower() in VALID_IMAGE_EXTENSIONS
    ]
    image_files.sort()
    if not image_files:
        raise ValueError(f"No image files found in: {images_path}")

    return len(image_files)


def parse_positive_int(value, flag_name):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{flag_name} must be > 0.")

    return parsed


def build_dense_view_indices(image_count, divisor, offset=1):
    if divisor is None:
        return None

    divisor = parse_positive_int(divisor, "--dense_view_divisor")
    offset = parse_positive_int(offset, "--dense_view_offset") - 1
    if offset >= divisor:
        raise ValueError("--dense_view_offset must be in [1, --dense_view_divisor].")

    dense_view_indices = list(range(offset, image_count, divisor))
    if not dense_view_indices:
        raise ValueError(
            f"No dense-view indices generated for image_count={image_count}, "
            f"divisor={divisor}, offset={offset + 1}."
        )

    return dense_view_indices


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate split-Nviews.json from a comma-separated training index list, "
            "optionally also emitting dense_view.json from evenly sampled all-image subsets."
        )
    )
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        required=True,
        help="Path to the scan directory, e.g. data/denseview/scan1",
    )
    parser.add_argument(
        "--train",
        "-train",
        dest="train",
        type=str,
        required=True,
        help="Comma-separated training view indices, e.g. 1,2,3,4,6,23",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Optional output json path. Defaults to <source_path>/split-<N>views.json",
    )
    parser.add_argument(
        "--dense_view_divisor",
        type=int,
        default=None,
        help=(
            "Also emit dense_view.json using every Nth image from the full image set. "
            "Use 2 / 3 / 4 for 1/2, 1/3, 1/4 sampling."
        ),
    )
    parser.add_argument(
        "--dense_view_output",
        type=str,
        default=None,
        help="Optional dense_view json path. Defaults to <source_path>/dense_view.json",
    )
    parser.add_argument(
        "--dense_view_offset",
        type=int,
        default=1,
        help=(
            "1-based phase for dense-view subsampling. 1 means indices 0,N,2N...; "
            "2 means 1,1+N,1+2N..."
        ),
    )
    args = parser.parse_args()

    train_indices = parse_train_indices(args.train)
    images_path = os.path.join(args.source_path, "images")
    image_count = count_images(images_path)

    invalid_indices = [idx for idx in train_indices if idx < 0 or idx >= image_count]
    if invalid_indices:
        raise ValueError(
            f"Training indices out of range [0, {image_count - 1}]: {invalid_indices}"
        )

    train_index_set = set(train_indices)
    test_indices = [idx for idx in range(image_count) if idx not in train_index_set]

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(
            args.source_path, f"split-{len(train_indices)}views.json"
        )

    split_config = {
        "train": train_indices,
        "test": test_indices,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(split_config, file, indent=4)
        file.write("\n")

    print(f"Saved split file to: {output_path}")
    print(f"Train views: {len(train_indices)}")
    print(f"Test views: {len(test_indices)}")

    dense_view_indices = build_dense_view_indices(
        image_count,
        args.dense_view_divisor,
        args.dense_view_offset,
    )
    if dense_view_indices is not None:
        dense_view_output = args.dense_view_output
        if dense_view_output is None:
            dense_view_output = os.path.join(args.source_path, "dense_view.json")

        dense_view_config = {
            "train": dense_view_indices,
        }

        with open(dense_view_output, "w", encoding="utf-8") as file:
            json.dump(dense_view_config, file, indent=4)
            file.write("\n")

        divisor = args.dense_view_divisor
        offset = args.dense_view_offset
        print(f"Saved dense-view file to: {dense_view_output}")
        print(
            f"Dense-view sampling: every {divisor} image(s), "
            f"offset {offset} -> {len(dense_view_indices)} views"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
