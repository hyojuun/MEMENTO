#!/bin/bash

# Add project directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/HabitatLLM"

# Get current date and time for logging
now=$(date +%m-%d_%H-%M)

# Default paths
INPUT_FILE="data/datasets/PEAD/v1/v1_val_stage2_filtered.json"
OUTPUT_FILE="data/datasets/PEAD/v1/v1_val_stage2_filtered_combined.json"

# Usage function
function print_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input PATH    Input file path (default: $INPUT_FILE)"
    echo "  -o, --output PATH   Output file path (default: $OUTPUT_FILE)"
    echo "  -h, --help          Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo "Starting episode combination at ${now}"
echo "Input file: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"

# Run the Python script with the specified arguments
python src/combine_episode/combine_episodes.py --input "${INPUT_FILE}" --output "${OUTPUT_FILE}"

echo "Episode combination completed!" 