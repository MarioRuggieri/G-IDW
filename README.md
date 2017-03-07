# Parallel GPU Inverse Distance Weighting
A parallel GPU version of the IDW interpolation algorithm [1]

Language: C, Framework: CUDA 7.0 

Software inspired by [2]

# Authors
Mario Ruggieri

e-mail: mario.ruggieri@studenti.uniparthenope.it

Livia Marcellino

e-mail: livia.marcellino@uniparthenope.it
  
# Installation and Usage 

`src/demo.cu` is an example of usage on random data. It launches 25 times the CPU and GPU algorithms and shows for both the mean time and standard deviation.

**Demo installation**
  ```
  cd src
  make
  ```
In this way you are making `demo` binary file
	
**Demo usage**

* 1th argument: number of known points

* 2th argument: number of values to interpolate

* 3th argument : number of CUDA block threads

Example:

	./demo 10000 1000 100

# Version
This is a beta version made for academic purposes.
	
# Licensing
Please read LICENSE file.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

# References
[1] Shepard D. , A two-dimensional interpolation function for irregularly-spaced data, Proceedings of the 1968 ACM National Conference. pp. 517–524 

[2] Hennebohl K., Appel M., Pebesma E. , Spatial Interpolation in Massively Parallel Computing Environments, Institute for Geoinformatics, University of Muenster, 2011 

