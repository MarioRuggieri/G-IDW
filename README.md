# Parallel GPU Inverse Distance Weighting
A parallel GPU version of the IDW interpolation algorithm [1]

Language: C, Framework: CUDA 8.0 

# Authors
Mario Ruggieri

e-mail: mario.ruggieri@uniparthenope.it

Livia Marcellino

e-mail: livia.marcellino@uniparthenope.it
  
# Installation and Usage 

`src/demo.cu` is an example of usage on random data. It launches 25 times the CPU and GPU algorithms and shows for both the average execution time and standard deviation.

**Demo installation**
  ```
  cd src
  make
  ```
In this way you are making `demo` binary file
	
**Demo usage**

* 1th argument: type of usage (1 = file, 2 = random data)

* 2th argument: (type 1) known 3D points dataset file / (type 2) number of known values

* 3th argument : (type 1) 2D locations file / (type 2) number of values to interpolate

* 4th argument : number of CUDA block threads

* 5th argument : search radius

Examples:

	./demo 1 dataset.txt grid.txt 256 400
	./demo 2 20000 300000 256 400
	
CPU and GPU output are saved into the current directory.

Each line of the dataset file must have this layout: longitude;latitude;z;

For example:

	13.70104167;40.84895833;-702.20;
	13.70312500;40.84895833;-700.20;
	...

Each line of the grid file (query locations) must have this layout: longitude;latitude;

For example:

	13.70104167;40.55104167;
	13.70154225;40.55104167;
	...
	
Software uses single-precision floating-points.

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
[1] L. Marcellino, R. Montella, S. Kosta, A. Galletti, D. Di Luccio, V. Santopietro, M. Ruggieri, M. Lapegna, L. D’Amore and G. Laccetti. Using GPGPU accelerated interpolation algorithms for marine bathymetry processing with on premises and cloud based computational resources. In 12th International Conference on Parallel Processing and Applied Mathematics, 2017, accepted

[2] Shepard D. , A two-dimensional interpolation function for irregularly-spaced data, Proceedings of the 1968 ACM National Conference. pp. 517–524 
