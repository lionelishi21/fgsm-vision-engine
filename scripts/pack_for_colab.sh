#!/bin/bash

# pack_for_colab.sh
# Packages the necessary FGSM code and data for Google Colab training.

echo "Packaging FGSM for Google Colab..."

# Create a temporary directory
TEMP_DIR="fgsm_colab_package"
mkdir -p $TEMP_DIR

# Copy essential files
cp -r src $TEMP_DIR/
cp -r data $TEMP_DIR/
cp -r configs $TEMP_DIR/ 2>/dev/null || true
cp -r scripts $TEMP_DIR/
cp requirements.txt $TEMP_DIR/

# Create the zip archive
ZIP_NAME="fgsm_colab_package.zip"
zip -r $ZIP_NAME $TEMP_DIR/

# Cleanup
rm -rf $TEMP_DIR

echo "Packaging complete!"
echo "Please upload '$ZIP_NAME' to the root of your Google Drive."
