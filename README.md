# AI-image-Detector

To setup conda environment, download miniconda and run:

    conda env create -f environment.yml
    
    conda activate KandidatENV

To add dependencies, run:

    conda install <package>
    
    conda env export --from-history > environment.yml

If you are missing a dependency, run:

    conda env update --file environment.yml --prune
