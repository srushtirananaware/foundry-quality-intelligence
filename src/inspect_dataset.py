from pathlib import Path

# Location of the dataset
DATASET_PATH = Path("data/raw/casting_data/casting_data")

# Classes in the dataset
classes = ["def_front", "ok_front"]

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

print("=" * 50)
print("CASTING DATASET INSPECTION")
print("=" * 50)

for split in ["train", "test"]:
    split_path = DATASET_PATH / split

    print(f"\n{split.upper()} DATASET")
    print("-" * 30)

    total = 0

    for class_name in classes:
        class_path = split_path / class_name

        images = [
            file for file in class_path.iterdir()
            if file.suffix.lower() in image_extensions
        ]

        count = len(images)
        total += count

        print(f"{class_name}: {count}")

    print(f"Total: {total}")

print("\n" + "=" * 50)
print("DATASET INSPECTION COMPLETE")
print("=" * 50)