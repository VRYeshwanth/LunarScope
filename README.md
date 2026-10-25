# LunarScope

## Dataset Installation Guide

1. Download the dataset from HuggingFace
```bash
hf download LLukaoo/LANYUE-bench archives/MSLCD.tar --repo-type dataset --local-dir .
```

2. Create a new folder named 'data' and extract the MSLCD.tar file
```bash
mkdir data

tar -xf archives/MSLCD.tar -C data
```