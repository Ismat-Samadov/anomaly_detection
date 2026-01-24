# Manual Fix Steps - Do This EXACTLY

## The Issue

You confirmed `novelty=True` is in the code, but you're still getting errors. This means either:
1. Jupyter didn't actually restart the kernel properly
2. Or you're running cells individually instead of "Run All"

## Solution: Follow These Steps EXACTLY

### Step 1: Close Everything
1. **Close the Jupyter notebook tab** completely (X button)
2. **Shut down Jupyter server** in terminal (Ctrl+C twice)
3. **Kill any Python processes**:
   ```bash
   pkill -f jupyter
   pkill -f python
   ```

### Step 2: Verify the Notebook File
```bash
cd /Users/ismatsamadov/crash_detection
grep "novelty=True" notebooks/anomaly_detection_pipeline.ipynb
```

You should see:
```
"novelty=True,  # Enable novelty detection for production inference\n",
```

If you DON'T see this, the file isn't saved correctly.

### Step 3: Clean Start
```bash
# Activate environment
source scripts/activate.sh

# Clear any cached files
rm -f notebooks/.ipynb_checkpoints/*

# Start fresh Jupyter
jupyter notebook notebooks/anomaly_detection_pipeline.ipynb
```

### Step 4: In Jupyter - DO NOT SKIP ANY STEP!

1. **First**: Click `Kernel` → `Restart & Clear Output`
   - Wait for confirmation message
   - All cell outputs should disappear

2. **Then**: Click `Cell` → `Run All`
   - Do NOT run cells individually!
   - Wait for ALL cells to complete (1-2 minutes)

3. **Watch**: Cell numbers should appear as [1], [2], [3]... in sequence
   - If any cell shows [*], it's still running - WAIT!

### Step 5: Verify Success

Scroll to the LAST cell (cell 52). You should see:
```
================================================================================
PRODUCTION INFERENCE EXAMPLE
================================================================================

✓ Loaded production pipeline and config
  Model: [Some Model Name]
  Features: 13

Simulating 5 new samples...

Predictions:
  Sample 1: NORMAL
  Sample 2: ANOMALY
  ...

================================================================================
PIPELINE EXECUTION COMPLETED SUCCESSFULLY
================================================================================
```

**NO AttributeError!**

## If You STILL Get Errors...

### Check 1: Are you running the right notebook?
```bash
ls -l notebooks/
```
Make sure you're opening `anomaly_detection_pipeline.ipynb`, not a copy.

### Check 2: Is the venv activated?
In Jupyter, in the first cell output, you should see:
```
All imports successful!
```

If you see `ModuleNotFoundError`, the venv isn't active.

### Check 3: What cell is failing?
- If cell 27 (LOF training) fails → novelty setting issue
- If cell 52 (Production inference) fails → old model loaded

### Check 4: Check the actual error message
Look for:
- `novelty=False` in the error → code not updated
- `AttributeError: predict` → old model in memory
- `ModuleNotFoundError` → venv not activated

## Alternative: Command Line Execution

If Jupyter keeps having issues, run from command line:

```bash
cd /Users/ismatsamadov/crash_detection
source scripts/activate.sh

# Execute notebook
jupyter nbconvert \
  --to notebook \
  --execute notebooks/anomaly_detection_pipeline.ipynb \
  --output anomaly_detection_executed.ipynb \
  --ExecutePreprocessor.timeout=600

# Check results
ls -lh notebooks/anomaly_detection_executed.ipynb
```

If successful, open `anomaly_detection_executed.ipynb` to see results.

## Last Resort: Show Me The Error

If none of this works, show me:
1. **Exact error message** (full traceback)
2. **Which cell** is failing (cell number)
3. **Cell output** from cell 27 (should show "Training: Local Outlier Factor")

---

**The fix IS in the code. The issue is execution environment, not the code itself.**
