# ---------------------------------------------------------------------------
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License version 2 as
#  published by the Free Software Foundation.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# ---------------------------------------------------------------------------

# Author: Gabriele Mencagli <mencagli@di.unipi.it>
# Date:   June 2017

all:
	$(MAKE) -C src

sum_test_cpu:
	$(MAKE) sum_test_cpu -C src

sum_test_gpu:
	$(MAKE) sum_test_gpu -C src

spatial_query:
	$(MAKE) spatial_query -C src

clean:
	$(MAKE) clean -C src

.DEFAULT_GOAL := all
.PHONY: all sum_test_cpu sum_test_gpu spatial_query clean