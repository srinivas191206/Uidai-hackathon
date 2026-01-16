# Deployment Guide: UIDAI Analytics Command Center

This document provides the exact steps to deploy your dashboard to the **srinivas1912006/uidai** Hugging Face Space.

## 1. Prerequisites
- **Hugging Face Access Token**: Generate one with **Write** permissions at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).
- **Hugging Face CLI**: Installed and ready.

## 2. Pushing to Hugging Face Space
Run these commands in your terminal inside the `/Users/thaladasrinivas/Desktop/UIDAI` directory:

```bash
# 1. Initialize git if not already done
git init

# 2. Add Hugging Face Space as a remote
git remote add hf https://huggingface.co/spaces/srinivas1912006/uidai

# 3. Add your files
git add .

# 4. Commit your changes
git commit -m "Deploy UIDAI Analytics Command Center"

# 5. Push to Hugging Face
# When prompted for your username, use your HF username.
# When prompted for a password, PASTE your Access Token.
git push hf master --force
```

## 3. Handling Large Files (LFS)
Since `enrolment_data_main.csv` is ~44MB, using Git LFS is recommended for faster pushes and better storage:

```bash
# Install Git LFS
git lfs install

# Track the large CSV file
git lfs track "*.csv"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for CSV files"
```

## 4. Troubleshooting
- **Port Error**: Ensure `Dockerfile` still has `EXPOSE 7860` and the CMD uses `--server.port=7860`.
- **Large File Error**: If the push fails because of file size, follow the Git LFS steps above.
- **Authentication**: If you get a 403 error, ensure your Access Token has **Write** permissions.
