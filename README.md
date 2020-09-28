# EIM Package ( Empirical Interpolation Method) 

This packages is the implmentation of the [EIM (Empirical Interpolation Method) algorithm](https://hal.archives-ouvertes.fr/hal-00174797/document). This algorithm is a dimensionality reduction algorithm. It can be used to preprocess some data before creating regression or classification models or any type of machine learning algo based on data. The main advantage of eim is that we can choose a desired precision is the compresion of the data information. The idea is to project each lines in an reduced interpolation based with is compute with magic points that are choosen as the worst interpolate points and correct at each new step of the algo.




The algorithm has been first introduce in this [2007 paper](https://hal.archives-ouvertes.fr/hal-00174797/document) : 

thanks to :

Yvon Maday, Ngoc Cuong Nguyen, Anthony T. Patera, George S.H. Pau.  A general, multipurposeinterpolation procedure: the magic points. 2007. l-00174797

* To install and uses the library

```shell
pip install dist/eim_samsja_faycal-0.0.1-py3-none-any.whl
```

* To build the packages from sources

```shell
python setup.py sdist bdist_wheel
```




## About the project

This packages has been developed in the context of a research project at the [University Of Technology Of Compiègne (UTC)](https://www.utc.fr/) in France. 

### Authors

* **Sami Jaghouar** ( my repo)
* **Faycal rekbi**

### Contributor and Supervisor

* **Florian De Vuyst** : head of LMAC ( [laboratory of Applied Mathematics of Compiègne](http://lmac.utc.fr/ )

* **Anne-Virginie Salsac** : CNRS Director of Research at BMBI ( [laboraty of biomecanics et bioengineering](https://bmbi.utc.fr/))

* **Claire Dupont**

