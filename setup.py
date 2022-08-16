from setuptools import find_packages, setup

'''
@Author: Pedro Herrera-Lormendez
'''

vfile = {}
exec(open('jcclass/version.py').read(), vfile)

with open('docs/README_pypi.md', 'r') as fh:
	long_description = fh.read()

#with open('jc_class/requirements.txt', 'r') as fh:
#	install_requires = fh.read()


setup(
	name='jcclass',
	include_package_data = True,
	keywords='circulations, CTs, WTs, synoptic',
	author = 'Pedro Herrera-Lormendez',
	author_email='peth31@gmail.com',
	description = 'Jenkinson and Collison automated gridded classification',
	long_description = long_description,
    long_description_content_type='text/markdown',	
	url = 'https://github.com/PedroLormendez/jc_module',
	license = 'MIT',
	packages =['jcclass'],
	zip_safe = False,
	python_requires='>=3.6',
	setuptools_git_versioning ={
		'version_callback' : vfile['__version__'],
		'dev_template'  : '{tag}.posst{ccount}',
		'dirty_template': '{tag}.post{ccount}'
		},
	setup_requires = ['setuptools-git-versioning', 'numpy'],
	install_requires = [
						'numpy>=1.19.2',
						'xarray>=0.16.2',
						'matplotlib>=3.2.0',
						'cartopy>=0.17.0',
						'cftime',
						'netCDF4'
	]
	)