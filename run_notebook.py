#!/usr/bin/env python3
"""
Run the notebook programmatically to ensure clean execution.
"""

import subprocess
import sys
from pathlib import Path

notebook_path = Path(__file__).parent / "notebooks" / "anomaly_detection_pipeline.ipynb"
output_path = Path(__file__).parent / "notebooks" / "anomaly_detection_pipeline_executed.ipynb"

print("="*70)
print("EXECUTING NOTEBOOK WITH CLEAN ENVIRONMENT")
print("="*70)
print(f"\nNotebook: {notebook_path}")
print(f"Output: {output_path}")
print("\nThis will take 1-2 minutes...\n")

try:
    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            str(notebook_path),
            "--output", str(output_path),
            "--ExecutePreprocessor.timeout=600"
        ],
        check=True,
        capture_output=True,
        text=True
    )

    print("✅ NOTEBOOK EXECUTED SUCCESSFULLY!")
    print(f"\nOutput saved to: {output_path}")
    print("\nYou can now open the executed notebook to see all results.")

except subprocess.CalledProcessError as e:
    print("❌ EXECUTION FAILED!")
    print("\nError output:")
    print(e.stderr)
    print("\nStandard output:")
    print(e.stdout)
    sys.exit(1)
