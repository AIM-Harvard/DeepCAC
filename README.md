# DeepCAC

Fully automatic coronary calcium risk assessment using Deep Learning. This 
work was published in 
[nature communications](https://doi.org/10.1038/s41467-021-20966-2) 
    - Open access!

If you use code or parts of this code in your work, please cite our 
publication:  
*Zeleznik, R., Foldyna, B., Eslami, P. et al. Deep convolutional neural 
networks to predict cardiovascular risk from computed tomography.
Nat Commun 12, 715 (2021). https://doi.org/10.1038/s41467-021-20966-2*

[<img src="https://media.springernature.com/full/nature-cms/uploads/product/ncomms/header-03d2e325c0a02f6df509e5730e9be304.svg">](https://doi.org/10.1038/s41467-021-20966-2)


## Repository Structure

The DeepCAC repository is structured as follows:

* All the source code to run the deep-learning-based fully automatic coronary calcium risk assessment pipeline is found under the `src` folder.
* Four sample subjects' CT data and the associated manual segmentation masks (heart + CAC), as well as all the models weights necessary to run the pipeline, are stored under the `data` folder.
* AI and manual results for the NLST cohort in the paper as well as the statistical analysis are located in the `stats` folder.

Additional details on the content of the subdirectories and their structure can be found in the markdown files stored in the former.

## Setup

This code was developed and tested using Python 2.7.17 on Ubuntu 18.04 with Cuda 10.1 and libcudnn 7.6.

For the code to run as intended, all the packages under `requirements.txt` should be installed. In order not to break previous installations and ensure full compatibility, it's highly recommended to create a virtual environment to run the DeepCAC pipeline in. Here follows an example of set-up using `python virtualenv`:

```
# install python's virtualenv
sudo pip install virtualenv

# parse the path to the python2 interpreter
export PY2PATH=$(which python2)

# create a virtualenv with such python2 interpreter named "venv"
# (common name, already found in .gitignore)
virtualenv -p $PY2PATH venv 

# activate the virtualenv
source venv/bin/activate
```

At this point, `(venv)` should be displayed at the start of each bash line. Furthermore, the command `which python2` should return a path similar to `/path/to/folder/venv/bin/python2`. Once the virtual environment is activated:

```
# once the virtualenv is activated, install the dependencies
pip install -r requirements.txt
```

At this stage, everything should be ready for the data to be processed by the DeepCAC pipeline. Additional details can be found in the markdown file under `src`.

The virtual environment can be deactivated by running:

```
deactivate
```

## Acknowledgements

Code development: RZ <br>
Code testing, refactoring and documentation: DB

## Disclaimer

The code and data of this repository are provided to promote reproducible 
research. They are not intended for clinical care or commercial use.

The software is provided "as is", without warranty of any kind, express or 
implied, including but not limited to the warranties of merchantability, 
fitness for a particular purpose and noninfringement. In no event shall the 
authors or copyright holders be liable for any claim, damages or other 
liability, whether in an action of contract, tort or otherwise, arising 
from, out of or in connection with the software or the use or other 
dealings in the software.

## Example data

We include four randomy selected cases from the LIDC-IDRI dataset plus manual segmentations. The full dataset 
can be found at https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI. The data is distributed under the Creative Commons Attribution 3.0 Unported License
https://creativecommons.org/licenses/by/3.0/
license.
The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.

Armato III, SG; McLennan, G; Bidaut, L; McNitt-Gray, MF; Meyer, CR; Reeves, AP; Zhao, B; Aberle, DR; Henschke, CI; Hoffman, Eric A; Kazerooni, EA; MacMahon, H; van Beek, EJR; Yankelevitz, D; Biancardi, AM; Bland, PH; Brown, MS; Engelmann, RM; Laderach, GE; Max, D; Pais, RC; Qing, DPY; Roberts, RY; Smith, AR; Starkey, A; Batra, P; Caligiuri, P; Farooqi, Ali; Gladish, GW; Jude, CM; Munden, RF; Petkovska, I; Quint, LE; Schwartz, LH; Sundaram, B; Dodd, LE; Fenimore, C; Gur, D; Petrick, N; Freymann, J; Kirby, J; Hughes, B; Casteele, AV; Gupte, S; Sallam, M; Heath, MD; Kuhn, MH; Dharaiya, E; Burns, R; Fryd, DS; Salganicoff, M; Anand, V; Shreter, U; Vastagh, S; Croft, BY; Clarke, LP. (2015). Data From LIDC-IDRI. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX 
 
Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204

Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. (2013) The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, pp 1045-1057. DOI: https://doi.org/10.1007/s10278-013-9622-7
