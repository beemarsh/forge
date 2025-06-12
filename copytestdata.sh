#!/bin/sh

FOLDER_NAME=""

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --name=*)
            FOLDER_NAME="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Check if folder name was provided
if [ -z "$FOLDER_NAME" ]; then
    echo "Error: --name is required."
    echo "Usage: $0 --name=folder_name"
    exit 1
fi

FOLDER_NAME="./test_data/$FOLDER_NAME"

# Create the folder
mkdir -p "$FOLDER_NAME"

# Check if mkdir succeeded
if [ $? -ne 0 ]; then
    echo "Failed to create folder: $FOLDER_NAME"
    exit 1
fi

mv hostfiles "$FOLDER_NAME"
mv neox.* "$FOLDER_NAME"
mv .deepspeed_env "$FOLDER_NAME"
mv logs "$FOLDER_NAME"
mv tensorboard "$FOLDER_NAME"
cp ./job.sb "$FOLDER_NAME"

cp -r ./configs/ "$FOLDER_NAME"
rm -rf ./checkpoints/
