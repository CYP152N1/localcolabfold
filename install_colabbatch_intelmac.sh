#!/bin/bash

# check commands
type wget || { echo "wget command is not installed. Please install it at first using Homebrew." ; exit 1 ; }
type gsed || { echo "gnu-sed command is not installed. Please install it at first using Homebrew." ; exit 1 ; }
type hhsearch || { echo "hhsearch command is not installed. Please install it at first using Homebrew." ; exit 1 ; }
type kalign || { echo "kalign command is not installed. Please install it at first using Homebrew." ; exit 1 ; }

# check whether Apple Silicon (M1 mac) or Intel Mac
arch_name="$(uname -m)"

if [ "${arch_name}" = "x86_64" ]; then
    if [ "$(sysctl -in sysctl.proc_translated)" = "1" ]; then
        echo "Running on Rosetta 2"
    else
        echo "Running on native Intel"
    fi
elif [ "${arch_name}" = "arm64" ]; then
    echo "Running on Apple Silicon (M1 mac)"
    echo "This installer is only for intel Mac. Use install_colabfold_M1mac.sh to install on this Mac."
    exit 1
else
    echo "Unknown architecture: ${arch_name}"
    exit 1
fi

CURRENTPATH=`pwd`
COLABFOLDDIR="${CURRENTPATH}/colabfold_batch"

mkdir -p ${COLABFOLDDIR}
cd ${COLABFOLDDIR}
wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt --no-check-certificate
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ./Miniconda3-latest-MacOSX-x86_64.sh -b -p ${COLABFOLDDIR}/conda
rm Miniconda3-latest-MacOSX-x86_64.sh
. "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${COLABFOLDDIR}/conda/condabin:${PATH}"
conda create -p $COLABFOLDDIR/colabfold-conda python=3.7 -y
conda activate $COLABFOLDDIR/colabfold-conda
conda update -n base conda -y
conda install -c conda-forge python=3.7 openmm==7.5.1 pdbfixer -y
# patch to openmm
wget -qnc https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch --no-check-certificate
(cd ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages; patch -s -p0 < ${COLABFOLDDIR}/openmm.patch)
rm openmm.patch
# install ColabFold and Jaxlib
colabfold-conda/bin/python3.7 -m pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
colabfold-conda/bin/python3.7 -m pip install https://storage.googleapis.com/jax-releases/mac/jaxlib-0.1.74-cp37-none-macosx_10_9_x86_64.whl

# bin directory to run
mkdir -p $COLABFOLDDIR/bin
cd $COLABFOLDDIR/bin
cat << EOF > colabfold_batch
#!/bin/sh
$COLABFOLDDIR/colabfold-conda/bin/colabfold_batch --cpu \$@
EOF
chmod +x colabfold_batch

# hack to share the parameter files in a workstation.
gsed -i -e "s#props_path = \"stereo_chemical_props.txt\"#props_path = \"${COLABFOLDDIR}/stereo_chemical_props.txt\"#" ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold/batch.py
gsed -i -e "s#kalign_binary_path=\"kalign\"#kalign_binary_path=\"/usr/local/bin/kalign\"#g" ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold/batch.py
gsed -i -e "s#binary_path=\"hhsearch\"#binary_path=\"/usr/local/bin/hhsearch\"#g" ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold/batch.py
gsed -i -e "s#Path(appdirs.user_cache_dir(__package__ or \"colabfold\"))#${COLABFOLDDIR}#g" ${COLABFOLDDIR}/colabfold-conda/lib/python3.7/site-packages/colabfold/download.py

echo "Installation of colabFold_batch finished."
echo "Note: AlphaFold2 weight parameters will be donwloaded at ${COLABFOLDDIR}/params directory in the first run."
echo "Please set your PATH to ${COLABFOLDDIR}/bin to run 'colabfold_batch'."
echo "i.e. For Bash, export PATH=\"${COLABFOLDDIR}/bin:\$PATH\""
echo "For more details, please type 'colabfold_batch --help'."