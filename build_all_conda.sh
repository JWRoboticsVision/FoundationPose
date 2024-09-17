#!/bin/bash

# Get the root directory of the project
PROJ_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Function to log messages
log_message() {
    echo "[INFO] $1"
}

# Install mycpp
log_message "Building mycpp..."
MYCPP_DIR="${PROJ_ROOT}/mycpp"
BUILD_DIR="${MYCPP_DIR}/build"

if [ -d "$BUILD_DIR" ]; then
    log_message "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

log_message "Running CMake and Make..."
if cmake .. && make -j$(nproc); then
    log_message "mycpp built successfully."
else
    echo "[ERROR] Failed to build mycpp." >&2
    exit 1
fi

# Optional: Install mycuda (commented out)
# log_message "Installing mycuda..."
# MYCUDA_DIR="${PROJ_ROOT}/bundlesdf/mycuda"
# cd "$MYCUDA_DIR" || { echo "[ERROR] Failed to cd to $MYCUDA_DIR"; exit 1; }
# rm -rf build *egg* *.so
# if python -m pip install -e .; then
#     log_message "mycuda installed successfully."
# else
#     echo "[ERROR] Failed to install mycuda." >&2
#     exit 1
# fi

# Return to the project root directory
cd "$PROJ_ROOT" || { echo "[ERROR] Failed to cd to $PROJ_ROOT"; exit 1; }

log_message "Returned to project root directory."