# p-winds
 Python implementation of Parker wind models for planetary atmospheres.

Aims
----
The main objective of this code is to produce a simplified, 1-D model of the upper atmosphere of a planet.

Background
----------
`p-winds` is largely based on the theoretical framework of [Oklopčić & Hirata (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...855L..11O/abstract) and [Lampón et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..13L/abstract), which themselves based their work on the stellar wind model of [Parker (1958)](https://ui.adsabs.harvard.edu/abs/1958ApJ...128..664P/abstract).

Installation
------------
First, clone the repository:
```angular2html
git clone https://github.com/ladsantos/p-winds.git
```
And then navigate to it, and install it:
```angular2html
cd p-winds
python setup.py install
```