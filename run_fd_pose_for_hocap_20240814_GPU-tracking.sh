#!/bin/bash
# Author: Jikai Wang
# Email: jikai.wang@utdallas.edu

# Get the root directory of the project
PROJ_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

# Get the base path of Conda
if ! CONDA_PATH=$(conda info --base 2>/dev/null); then
    echo "Error: Conda is not installed or not available in your PATH."
    exit 1
fi

# Set the Python executable path from the Conda environment
PYTHON_PATH="${CONDA_PATH}/envs/foundationpose/bin/python"

# Check if the script file exists
SCRIPT_FILE="${PROJ_ROOT}/run_fd_pose_for_hocap_20240814.py"
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script file ${SCRIPT_FILE} does not exist."
    exit 1
fi

# Default GPU ID (can be overridden by passing a command-line argument)
GPU_ID=${1:-0}

# Function to run the script with given parameters
run_script() {
    local SEQUENCE_FOLDER=$1
    local OBJECT_IDX=$2
    local START_FRAME=$3
    local END_FRAME=$4

    CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_PATH}" "${SCRIPT_FILE}" \
        --sequence_folder "${PROJ_ROOT}/${SEQUENCE_FOLDER}" \
        --object_idx "${OBJECT_IDX}" \
        --start_frame "${START_FRAME}" \
        --end_frame "${END_FRAME}" \
        --running_mode "tracking"
}

# Define an associative array to map sequence folders to their respective parameters
declare -A SEQUENCES

# G05
SEQUENCES["data/HOCap/subject_1/20231025_170532"]="1 605 786; 2 215 216; 3 400 526; 4 20 171"
SEQUENCES["data/HOCap/subject_6/20231025_110646"]="1 360 541; 2 715 871; 3 1000 1241; 4 20 200"
SEQUENCES["data/HOCap/subject_6/20231025_110808"]="3 680 851; 4 30 211"

# G06
SEQUENCES["data/HOCap/subject_2/20231022_201942"]="1 90 331; 2 1380 1471; 3 500 791; 4 940 1191"
SEQUENCES["data/HOCap/subject_2/20231022_202115"]="1 370 641; 2 700 941; 3 1100 1351; 4 50 291"
SEQUENCES["data/HOCap/subject_6/20231025_111118"]="1 1040 1191; 2 330 501; 3 720 871; 4 40 175"
SEQUENCES["data/HOCap/subject_6/20231025_111357"]="1 1920 2356; 2 50 411; 3 620 971; 4 1200 1681"
SEQUENCES["data/HOCap/subject_8/20231024_180651"]="1 510 686; 2 740 916; 3 280 421; 4 50 201"
SEQUENCES["data/HOCap/subject_8/20231024_180733"]="1 830 1041; 2 50 271; 3 345 551; 4 605 790"

# G10
SEQUENCES["data/HOCap/subject_1/20231025_170231"]="1 575 801; 2 285 501; 3 885 1046; 4 25 211"

# G19
SEQUENCES["data/HOCap/subject_1/20231025_171117"]="1 30 201; 2 520 711; 3 260 451; 4 740 941"
SEQUENCES["data/HOCap/subject_2/20231023_163929"]="1 380 521; 2 940 1071; 3 50 201; 4 655 801"
SEQUENCES["data/HOCap/subject_2/20231023_164242"]="1 80 451; 2 1045 1466; 3 10 11; 4 560 901"
SEQUENCES["data/HOCap/subject_3/20231024_161937"]="1 640 851; 2 1005 1201; 3 60 201; 4 320 501"
SEQUENCES["data/HOCap/subject_3/20231024_162028"]="1 60 296; 2 460 681; 3 1250 1446; 4 840 1101"
SEQUENCES["data/HOCap/subject_5/20231027_113535"]="1 35 201; 2 695 811; 3 270 391; 4 455 611"

# G21
SEQUENCES["data/HOCap/subject_1/20231025_171314"]="1 435 566; 2 225 361; 3 30 171; 4 630 790"
SEQUENCES["data/HOCap/subject_1/20231025_171417"]="1 240 241; 2 680 820; 3 410 621; 4 15 221"
SEQUENCES["data/HOCap/subject_3/20231024_162756"]="1 40 176; 2 770 901; 3 530 661; 4 300 441"
SEQUENCES["data/HOCap/subject_3/20231024_162842"]="1 35 191; 2 530 711; 3 290 441; 4 820 1011"
SEQUENCES["data/HOCap/subject_4/20231026_164909"]="1 40 261; 2 30 31; 3 580 821; 4 320 541"
SEQUENCES["data/HOCap/subject_4/20231026_164958"]="1 40 321; 2 30 31; 3 780 1206; 4 400 741"
SEQUENCES["data/HOCap/subject_9/20231027_125315"]="1 920 1086; 2 640 791; 3 40 241; 4 350 531"
SEQUENCES["data/HOCap/subject_9/20231027_125407"]="1 1030 1271; 2 700 931; 3 30 266; 4 390 586"
SEQUENCES["data/HOCap/subject_9/20231027_125457"]="1 1460 1726; 2 1000 1335; 3 30 351; 4 505 901"

# Run the script for each sequence folder
for SEQUENCE_FOLDER in "${!SEQUENCES[@]}"; do
    PARAMS="${SEQUENCES[$SEQUENCE_FOLDER]}"  # Get the parameters as a string

    # Split PARAMS by semicolon and space to process each set individually
    IFS=';' read -ra PARAM_ARRAYS <<< "$PARAMS"

    for PARAM_SET in "${PARAM_ARRAYS[@]}"; do
        # Trim leading and trailing whitespace from PARAM_SET
        PARAM_SET=$(echo "$PARAM_SET" | xargs)
        
        # Split PARAM_SET into individual arguments and pass to run_script
        IFS=' ' read -r -a PARAM_ARRAY <<< "$PARAM_SET"
        run_script "$SEQUENCE_FOLDER" "${PARAM_ARRAY[@]}"
    done
done
