#!/bin/bash --login
#SBATCH --job-name=nucleo_run
#SBATCH --output=nucleo_runs/nucleo_run_%j.out
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=general-long

# Change to the scripts directory
cd /mnt/home/jdelker/_main/projects/osnap/scripts

# Export variables
export FLASH_MODELS=/mnt/research/SNAPhU/STIR/run_sukhbold
export SKYNET_OUTPUT=/mnt/home/jdelker/_main/projects/osnap/skynet_output
export NUCLEOSYNTH=/mnt/home/jdelker/_main/tools/nucleosynth
export PYTHONPATH=/mnt/home/jdelker/skynet_install/lib:${NUCLEOSYNTH}:/mnt/home/jdelker/_main/tools/flashbang:/mnt/home/jdelker/_main/projects/osnap:${PYTHONPATH}

# Load modules
module load GCCcore/12.3.0
module load zlib/1.2.13-GCCcore-12.3.0
module load binutils/2.40-GCCcore-12.3.0
module load GCC/12.3.0
module load numactl/2.0.16-GCCcore-12.3.0
module load XZ/5.4.2-GCCcore-12.3.0
module load libxml2/2.11.4-GCCcore-12.3.0
module load libpciaccess/0.17-GCCcore-12.3.0
module load hwloc/2.9.1-GCCcore-12.3.0
module load OpenSSL/1.1
module load libevent/2.1.12-GCCcore-12.3.0
module load PMIx/4.2.4-GCCcore-12.3.0
module load OpenMPI/4.1.5-GCC-12.3.0
module load gompi/2023a
module load ncurses/6.4-GCCcore-12.3.0
module load bzip2/1.0.8-GCCcore-12.3.0
module load cURL/8.0.1-GCCcore-12.3.0
module load libarchive/3.6.2-GCCcore-12.3.0
module load CMake
module load libreadline/8.2-GCCcore-12.3.0
module load Tcl/8.6.13-GCCcore-12.3.0
module load SQLite/3.42.0-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
module load Python
module load PCRE2/10.42-GCCcore-12.3.0
module load SWIG
module load gzip/1.12-GCCcore-12.3.0
module load lz4/1.9.4-GCCcore-12.3.0
module load zstd/1.5.5-GCCcore-12.3.0
module load ICU/73.2-GCCcore-12.3.0
module load Boost
module load Szip/2.1.1-GCCcore-12.3.0
module load HDF5
module load GSL

# Activate the osnap virtual environment
source /mnt/home/jdelker/.venv/bin/activate

# Verify we're using the correct Python
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"

# Get number of tracers from environment variable (default to 100 if not set)
echo "Running with mass of ${MASS}, alpha of ${ALPHA}, and ${NUM_TRACERS} tracers"

# Run the script
python quick_nucleo.py -f -t ${NUM_TRACERS} -a ${ALPHA} -m ${MASS}

