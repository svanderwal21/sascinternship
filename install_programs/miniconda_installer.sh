mkdir -p /exports/sascstudent/svanderwal2/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /exports/sascstudent/svanderwal2/miniconda3/miniconda.sh
bash /exports/sascstudent/svanderwal2/miniconda3/miniconda.sh -b -u -p /exports/sascstudent/svanderwal2/miniconda3
rm -rf /exports/sascstudent/svanderwal2/miniconda3/miniconda.sh
/exports/sascstudent/svanderwal2/miniconda3/bin/conda init bash
/exports/sascstudent/svanderwal2/miniconda3/bin/conda init zsh
