#!/bin/bash

type wget || { echo "wget command is not installed. Please install it at first using apt or yum." ; exit 1 ; }
type curl || { echo "curl command is not installed. Please install it at first using apt or yum. " ; exit 1 ; }

CURRENTPATH=`pwd`
COLABFOLDDIR="${CURRENTPATH}/colabfold_batch"

mkdir -p ${COLABFOLDDIR}
cd ${COLABFOLDDIR}
wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt --no-check-certificate
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${COLABFOLDDIR}/conda
rm Miniconda3-latest-Linux-x86_64.sh
. "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${COLABFOLDDIR}/conda/condabin:${PATH}"
conda create -p $COLABFOLDDIR/colabfold-conda python=3.7 -y
conda activate $COLABFOLDDIR/colabfold-conda
conda update -n base conda -y
conda install -c conda-forge python=3.7 cudnn==8.2.1.32 cudatoolkit==11.1.1 openmm==7.5.1 pdbfixer -y
# patch to openmm
wget -qnc https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch --no-check-certificate
(cd ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages; patch -s -p0 < ${COLABFOLDDIR}/openmm.patch)
rm openmm.patch
# install alignment tools
conda install -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0 -y
# install ColabFold and Jaxlib
colabfold-conda/bin/python3.7 -m pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
colabfold-conda/bin/python3.7 -m pip install https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.72+cuda111-cp37-none-manylinux2010_x86_64.whl

# bin directory to run
mkdir -p $COLABFOLDDIR/bin
cd $COLABFOLDDIR/bin
cat << EOF > colabfold_batch
#!/bin/sh
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
$COLABFOLDDIR/colabfold-conda/bin/colabfold_batch \$@
EOF
chmod +x colabfold_batch

# hack to share the parameter files in a workstation.
cd ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold
sed -i -e "s#props_path = \"stereo_chemical_props.txt\"#props_path = \"${COLABFOLDDIR}/stereo_chemical_props.txt\"#" batch.py
sed -i -e "s#kalign_binary_path=\"kalign\"#kalign_binary_path=\"${COLABFOLDDIR}/colabfold-conda/bin/kalign\"#g" ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold/batch.py
sed -i -e "s#binary_path=\"hhsearch\"#binary_path=\"${COLABFOLDDIR}/colabfold-conda/bin/hhsearch\"#g" ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold/batch.py
sed -i -e "s#Path(appdirs.user_cache_dir(__package__ or \"colabfold\"))#\"${COLABFOLDDIR}\"#g" download.py
cd ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/alphafold/relax
sed -i -e 's/CPU/CUDA/g' amber_minimize.py

echo "Installation of colabFold_batch finished."
echo "Note: AlphaFold2 weight parameters will be donwloaded at ${COLABFOLDDIR}/params directory in the first run."
echo "Please set your PATH to ${COLABFOLDDIR}/bin to run 'colabfold_batch'."
echo "i.e. For Bash, export PATH=\"${COLABFOLDDIR}/bin:\$PATH\""
echo "For more details, please type 'colabfold_batch --help'."