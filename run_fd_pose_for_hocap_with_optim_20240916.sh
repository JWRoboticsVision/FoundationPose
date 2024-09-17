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

# Check if the Python executable exists
if [ ! -x "$PYTHON_PATH" ]; then
    echo "Error: Python executable ${PYTHON_PATH} not found or is not executable."
    exit 1
fi

# Check if the script file exists
SCRIPT_FILE="${PROJ_ROOT}/run_fd_pose_for_hocap_with_optim_20240916.py"
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script file ${SCRIPT_FILE} does not exist."
    exit 1
fi

# Default GPU ID (can be overridden by passing a command-line argument)
GPU_ID=${1:-0}

# Function to run the script with given parameters
run_script() {
    local sequence_folder=$1
    local object_idx=$2

    CUDA_VISIBLE_DEVICES=${GPU_ID} "$PYTHON_PATH" "$SCRIPT_FILE" \
        --sequence_folder "$sequence_folder" \
        --object_idx "$object_idx"
}

OBJECT_IDS=(1 2 3 4)

ALL_SEQUENCES=(
    data/HOCap/subject_1/20231025_165502
    data/HOCap/subject_1/20231025_165807
    data/HOCap/subject_1/20231025_170105
    data/HOCap/subject_1/20231025_170231
    data/HOCap/subject_1/20231025_170532
    data/HOCap/subject_1/20231025_170650
    data/HOCap/subject_1/20231025_170959
    data/HOCap/subject_1/20231025_171117
    data/HOCap/subject_1/20231025_171314
    data/HOCap/subject_1/20231025_171417
    data/HOCap/subject_2/20231022_200657
    data/HOCap/subject_2/20231022_201316
    data/HOCap/subject_2/20231022_201449
    data/HOCap/subject_2/20231022_201556
    data/HOCap/subject_2/20231022_201942
    data/HOCap/subject_2/20231022_202115
    data/HOCap/subject_2/20231022_202617
    data/HOCap/subject_2/20231022_203100
    data/HOCap/subject_2/20231023_163929
    data/HOCap/subject_2/20231023_164242
    data/HOCap/subject_2/20231023_164741
    data/HOCap/subject_2/20231023_170018
    data/HOCap/subject_3/20231024_154531
    data/HOCap/subject_3/20231024_154810
    data/HOCap/subject_3/20231024_155008
    data/HOCap/subject_3/20231024_161209
    data/HOCap/subject_3/20231024_161306
    data/HOCap/subject_3/20231024_161937
    data/HOCap/subject_3/20231024_162028
    data/HOCap/subject_3/20231024_162327
    data/HOCap/subject_3/20231024_162409
    data/HOCap/subject_3/20231024_162756
    data/HOCap/subject_3/20231024_162842
    data/HOCap/subject_4/20231026_162155
    data/HOCap/subject_4/20231026_162248
    data/HOCap/subject_4/20231026_163223
    data/HOCap/subject_4/20231026_164131
    data/HOCap/subject_4/20231026_164812
    data/HOCap/subject_4/20231026_164909
    data/HOCap/subject_4/20231026_164958
    data/HOCap/subject_5/20231027_112303
    data/HOCap/subject_5/20231027_113202
    data/HOCap/subject_5/20231027_113535
    data/HOCap/subject_6/20231025_110646
    data/HOCap/subject_6/20231025_110808
    data/HOCap/subject_6/20231025_111118
    data/HOCap/subject_6/20231025_111357
    data/HOCap/subject_6/20231025_112229
    data/HOCap/subject_6/20231025_112332
    data/HOCap/subject_6/20231025_112546
    data/HOCap/subject_7/20231022_190534
    data/HOCap/subject_7/20231022_192832
    data/HOCap/subject_7/20231022_193506
    data/HOCap/subject_7/20231022_193630
    data/HOCap/subject_7/20231022_193809
    data/HOCap/subject_7/20231023_162803
    data/HOCap/subject_7/20231023_163653
    data/HOCap/subject_8/20231024_180111
    data/HOCap/subject_8/20231024_180651
    data/HOCap/subject_8/20231024_180733
    data/HOCap/subject_8/20231024_181413
    data/HOCap/subject_9/20231027_123403
    data/HOCap/subject_9/20231027_123725
    data/HOCap/subject_9/20231027_123814
    data/HOCap/subject_9/20231027_124057
    data/HOCap/subject_9/20231027_124926
    data/HOCap/subject_9/20231027_125019
    data/HOCap/subject_9/20231027_125315
    data/HOCap/subject_9/20231027_125407
    data/HOCap/subject_9/20231027_125457
)

log_file="${PROJ_ROOT}/run_fd_pose_for_hocap_with_optim_20240916.log"

# Add a timestamp to the log file
{
    echo "-------------------------------------------------------------------------------"
    echo "Log Date: $(date)"
    echo "-------------------------------------------------------------------------------"
} >> "$log_file"

# Run the script for each sequence and object ID
for SEQUENCE in "${ALL_SEQUENCES[@]}"; do
    {
        echo "###############################################################################"
        echo "# Run FoundationPose on ${SEQUENCE}"
        echo "###############################################################################"
    } >> "$log_file"

    for OBJ_ID in "${OBJECT_IDS[@]}"; do
        start_time=$(date +%s)
        echo "###############################################################################"
        echo "# Run FoundationPose on ${SEQUENCE} for object ${OBJ_ID}"
        echo "###############################################################################"
        run_script "$SEQUENCE" "$OBJ_ID"
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        # Log the output
        {
            echo "  - Object ID: ${OBJ_ID}"
            echo "    ** Execution time: ${duration} seconds"
        } >> "$log_file"
    done
done
