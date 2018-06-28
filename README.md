# Analysis of an allelic series using transcriptomic phenotypes

## David Angeles Albores & Paul W. Sternberg

This repo contains all the analyses for our manuscript [Analysis of an allelic series using transcriptomic phenotypes](https://www.biorxiv.org/content/early/2018/01/29/210724).

The folder structure is as follows:
```
docs         - Files for building the website
input        - Contains most of the required inputs into the supplementary. The exception are the processed Sleuth files.
output       - Figures output from running the code
kallisto     - Contains the processed TPM files and run information that was input into Sleuth. NOTE: The kallisto files contain a lot of extraneous RNA-seq data to this project, which was used to perform accurate batch corrections and improve statistical power
sleuth       - Contains the necessary scripts and input information to run Sleuth.
             - sleuth_strains - This folder contains the output from running sleuth
tex          - Everything that is required to assemble the final manuscript.
useful_notes - Some notes on Ras biology
```

Feel free to contact dangeles@caltech.edu or pws@caltech.edu if you have questions pertaining to this repository. 
