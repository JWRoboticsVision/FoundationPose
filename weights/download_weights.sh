#!/bin/bash

CURR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Download the folder '2023-10-28-18-33-37' from Google Drive
# https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC?usp=drive_link
mkdir -p ${CURR_DIR}/2023-10-28-18-33-37
gdown "https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC" --folder -O ${CURR_DIR}/2023-10-28-18-33-37

# Download the folder '2024-01-11-20-02-45' from Google Drive
# https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj?usp=drive_link
mkdir -p ${CURR_DIR}/2024-01-11-20-02-45
gdown "https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj" --folder -O ${CURR_DIR}/2024-01-11-20-02-45
