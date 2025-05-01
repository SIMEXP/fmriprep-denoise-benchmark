#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def convert_age(age_str):
    """
    Convert an age string of the form '35y8m26d' to a float representing years.
    If the age is already numeric or cannot be parsed, returns None.
    """
    if pd.isna(age_str):
        return None
    # Try to convert directly to float (in case it's already numeric)
    try:
        return float(age_str)
    except ValueError:
        pass
    # Use regex to parse strings like "35y8m26d"
    match = re.match(r"(\d+)y(\d+)m(\d+)d", str(age_str))
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        days = int(match.group(3))
        # Approximate conversion: months to years and days to years
        return years + months / 12.0 + days / 365.25
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Convert age values in participants.tsv from string format (e.g., '35y8m26d') to numeric (years)."
    )
    parser.add_argument("participants_path", help="Path to the participants.tsv file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (if not provided, the input file is overwritten)",
        default=None,
    )
    args = parser.parse_args()

    # Read the participants file (assumed to be tab-separated)
    df = pd.read_csv(args.participants_path, sep="\t")
    print("Columns in file:", df.columns.tolist())

    # Identify the age column. Adjust the column name if needed.
    if "Âge" in df.columns:
        age_col = "Âge"
    elif "Age" in df.columns:
        age_col = "Age"
    else:
        print("No age column found. Exiting.")
        return

    # Convert the age column
    df[age_col] = df[age_col].apply(convert_age)

    # Determine the output file
    output_file = args.output if args.output else args.participants_path

    # Save the updated DataFrame
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Converted file saved to {output_file}")

if __name__ == "__main__":
    main()