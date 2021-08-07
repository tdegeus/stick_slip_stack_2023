import argparse
import GooseSLURM
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*', type=str)
parser.add_argument('-n', '--group', nargs=1, type=int, default=1)
parser.add_argument('-w', '--walltime', nargs=1, type=str, default='24h')
args = parser.parse_args()
assert np.all([os.path.isfile(file) for file in args.files])

slurm = '''
# print jobid
echo "SLURM_JOBID = ${{SLURM_JOBID}}"
echo ""

# for safety set the number of cores
export OMP_NUM_THREADS=1

# load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
    conda activate code_layers_E5v4
elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
    conda activate code_layers_s6g1
elif [[ "${{SYS_TYPE}}" == *S6g1* ]]; then
    conda activate code_layers_s6g1
else
    echo "Unknown SYS_TYPE ${{SYS_TYPE}}"
    exit 1
fi

{0:s}
'''

commands = ['stdbuf -o0 -e0 Run {0:s}'.format(file) for file in args.files]

args.group = 1
ngroup = int(np.ceil(len(commands) / args.group))
fmt = str(int(np.ceil(np.log10(ngroup))))

for group in range(ngroup):

    c = commands[group * args.group: (group + 1) * args.group]
    command = '\n'.join(c)
    command = slurm.format(command)

    jobname = ('Run-{0:0' + fmt + 'd}').format(group)

    sbatch = {
        'job-name': 'layers-' + jobname,
        'out': jobname + '.out',
        'nodes': 1,
        'ntasks': 1,
        'cpus-per-task': 1,
        'time': args.walltime,
        'account': 'flexlab-frictionlayers',
        'partition': 'serial',
    }

    open(jobname + '.slurm', 'w').write(GooseSLURM.scripts.plain(command=command, **sbatch))
