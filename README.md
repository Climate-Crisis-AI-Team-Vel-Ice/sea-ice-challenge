# Sea Ice Challenge
Kaggle & Data Visualisation challenge for Climate Crisis AI 2021 Hackathon developed by the great Vel Ice team

## 1- Installation
### Install the libgeos-dev library
#### linux:
```
sudo apt-get install libgeos-dev
```
#### MAC (I have not tested this)
based on https://stackoverflow.com/questions/42299352/installing-basemap-on-mac-python
1- some home brewing
```
brew install matplotlib
brew install numpy
brew install geos
brew install proj
```
2- Download Basemap source tar file (https://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/), untar it
3- Add "export GEOS_DIR=/usr/local/Cellar/geos/3.5.0/" to a new line in .bash_profile, and then reload it via:
```
source ~/.bash_profile
```
4- From within untarred Basemap directory:
```
python setup.py install
```
#### Windows
follow https://matplotlib.org/basemap/users/installing.html

### Python requirements
Make sure you have python 3 installed as the default version of python then in a command window or terminal run:
```
python -m pip install -r requirements.txt 
```