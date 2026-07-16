from pathlib import Path
from unstructured.partition.auto import partition
import traceback

# Change this if needed
file_path = Path("uploads/Tally Claude AI proposal.docx")

print("=" * 60)
print("Current Working Directory :", Path.cwd())
print("File Path                 :", file_path)
print("Absolute Path             :", file_path.resolve())
print("Exists                    :", file_path.exists())
print("Is File                   :", file_path.is_file())
print("Is Directory              :", file_path.is_dir())
print("=" * 60)

if not file_path.exists():
    print("❌ File does not exist.")
    exit()

try:
    print("📄 Partitioning document...")

    elements = partition(
        filename=str(file_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )

    print(f"✅ Successfully partitioned {len(elements)} elements.\n")

    for i, element in enumerate(elements[:10], 1):
        print(f"----- Element {i} -----")
        print(type(element).__name__)
        print(str(element)[:300])
        print()

except Exception as e:
    print("❌ Partition failed")
    traceback.print_exc()
