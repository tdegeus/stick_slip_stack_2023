#!/bin/bash
#SBATCH --job-name build_s6g1
#SBATCH --out build_s6g1.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --constraint s6g1
#SBATCH --account pcsl
#SBATCH --partition build

# print jobid
echo "SLURM_JOBID = ${SLURM_JOBID}"
echo ""

# change current directory to the location of the sbatch command
cd "${SLURM_SUBMIT_DIR}"

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "${SYS_TYPE}" == *E5v4* ]]; then
    conda activate code_layers_E5v4
    mydir="build_E5v4"
elif [[ "${SYS_TYPE}" == *s6g1* ]]; then
    conda activate code_layers_s6g1
    mydir="build_s6g1"
elif [[ "${SYS_TYPE}" == *S6g1* ]]; then
    conda activate code_layers_s6g1
    mydir="build_s6g1"
else
    echo "Unknown SYS_TYPE ${SYS_TYPE}"
    exit 1
fi

if [ -d "${mydir}" ]; then
    rm -r "${mydir}"
fi

mkdir "${mydir}"
cd "${mydir}"
echo "*" > .gitignore

cmake .. -DCMAKE_INSTALL_PREFIX:PATH="${CONDA_PREFIX}"
make
make install
cd ..

rm -r "${mydir}"
