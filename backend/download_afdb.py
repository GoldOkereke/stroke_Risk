import wfdb
import os

# Create data directory
os.makedirs("data/afdb", exist_ok=True)

# List of records in the AFDB dataset
# 23 records have actual signal files (excluding 00735 and 03665 which only have annotations)
records = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936",
    "05091", "05121", "05261", "06426", "06453", "06995", "07162",
    "07859", "07879", "07910", "08215", "08219", "08378", "08405",
    "08434", "08455"
]

print("Downloading AFDB dataset...")
for record in records:
    print(f"Downloading record {record}...")
    try:
        # Download signal and annotation files
        wfdb.dl_database('afdb', 'data/afdb', records=[record])
    except Exception as e:
        print(f"Error downloading {record}: {e}")

print("Download complete!")