# Sea Ice Challenge
Kaggle & Data Visualisation challenge for Climate Crisis AI 2021 Hackathon developed by the great Vel Ice team <br />

With Random Forests <br />
Overall RMSE:  2.6 <br />

With Neural Networks
Overall RMSE:  2.94 <br />

With XGBoost <br />
Overall RMSE for velocity magnitude:  4.24 <br />
Overall RMSE for velocity direction:  1.21 <br />
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
2- Download Basemap source tar file (https://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/), untar it <br />
3- Add "export GEOS_DIR=/usr/local/Cellar/geos/3.5.0/" to a new line in .bash_profile, and then reload it via: <br />
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
Make sure you have python 3 installed as the default version of python then in a command window or terminal run: <br />
```
python -m pip install -r requirements.txt 
```

## 2- Run the software
### Data import, manipulation and plotting
run sea_ice_data_plot.ipynb

### Data correlation and PCA analysis
run sea_ice_data_PCA_analysis.ipynb <br />
run sea_ice_data_visualization.ipynb 

### Training
sea_ice_data_ML_XGB_RF.ipynb


