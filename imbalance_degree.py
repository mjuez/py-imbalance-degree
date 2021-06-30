#!/usr/bin/env python
# Copyright (C) 2019  Mario Juez-Gil <mariojg@ubu.es>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Aknowledgements
# ---------------
# This work was partially supported by the Consejería de Educación of the 
# Junta de Castilla y León and by the European Social Fund with the 
# EDU/1100/2017 pre-doctoral grants; by the project TIN2015-67534-P 
# (MINECO/FEDER, UE) of the Ministerio de Economía Competitividad of the 
# Spanish Government and the project BU085P17 (JCyL/FEDER, UE) of the Junta de 
# Castilla y León both cofinanced from European Union FEDER funds.

import numpy as np
from math import sqrt, log

__author__ = "Mario Juez-Gil"
__copyright__ = "Copyright 2019, Mario Juez-Gil"
__credits__ = ["Mario Juez-Gil", "Álvar Arnaiz-González", 
               "Cesar Garcia-Osorio", "Carlos López-Nozal",
               "Juan J. Rodriguez"]
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Mario Juez-Gil"
__email__ = "mariojg@ubu.es"

def imbalance_degree(classes, distance="EU"):
    """
    Calculates the imbalance degree [1] of a multi-class dataset.
    This metric is an alternative for the well known imbalance ratio, which
    is only suitable for binary classification problems.
    
    Parameters
    ----------
    classes : list of int.
        List of classes (targets) of each instance of the dataset.
    distance : string (default: EU).
        distance or similarity function identifier. It can take the following
        values:
            - EU: Euclidean distance.
            - CH: Chebyshev distance.
            - KL: Kullback Leibler divergence.
            - HE: Hellinger distance.
            - TV: Total variation distance.
            - CS: Chi-square divergence.
        
    References
    ----------
    .. [1] J. Ortigosa-Hernández, I. Inza, and J. A. Lozano, 
            “Measuring the class-imbalance extent of multi-class problems,” 
            Pattern Recognit. Lett., 2017.
    """
    def _eu(_d, _e):
        """
        Euclidean distance from empirical distribution 
        to equiprobability distribution.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        distance value.
        """
        summ = np.vectorize(lambda p : pow(p - _e, 2))(_d).sum()
        return sqrt(summ)
        
    def _ch(_d, _e):
        """
        Chebyshev distance from empirical distribution 
        to equiprobability distribution.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        distance value.
        """
        dif = np.vectorize(lambda p : abs(p - _e))(_d)
        return dif.max()
    
    def _kl(_d, _e):
        """
        Kullback Leibler divergence from empirical distribution 
        to equiprobability distribution.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        distance value.
        """
        kl = lambda p : 0.0 if p == 0 else p * log(p/_e)
        return np.vectorize(kl)(_d).sum()
    
    def _he(_d, _e):
        """
        Hellinger distance from empirical distribution 
        to equiprobability distribution.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        distance value.
        """
        summ = np.vectorize(lambda p : pow((sqrt(p) - sqrt(_e)), 2))(_d).sum()
        return (1 / sqrt(2)) * sqrt(summ)
    
    def _tv(_d, _e):
        """
        Total variation distance from empirical distribution 
        to equiprobability distribution.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        distance value.
        """
        summ = np.vectorize(lambda p : abs(p - _e))(_d).sum()
        return (1 / 2) * summ
    
    def _cs(_d, _e):
        """
        Chi-square divergence from empirical distribution 
        to equiprobability distribution.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        distance value.
        """
        summ = np.vectorize(lambda p : pow((p - _e), 2) / _e)(_d).sum()
        return summ
    
    def _min_classes(_d, _e):
        """
        Calculates the number of minority classes. We call minority class to
        those classes with a probability lower than the equiprobability term.
        
        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).
        
        Returns
        -------
        Number of minority clases.
        """
        return len(_d[_d < _e])
    
    def _i_m(_K, _m):
        """
        Calculates the distribution showing exactly m minority classes with the
        highest distance to the equiprobability term. This distribution is 
        always the same for all distance functions proposed, and is explained
        in [1].
        
        Parameters
        ----------
        _K : int.
            The number of classes (targets).
        _m : int.
            The number of minority classes. We call minority class to
            those classes with a probability lower than the equiprobability 
            term.
        
        Returns
        -------
        A list with the i_m distribution.
        """
        min_i = np.zeros(_m)
        maj_i = np.ones((_K - _m - 1)) * (1 / _K)
        maj = np.array([1 - (_K - _m - 1) / _K])
        return np.concatenate((min_i, maj_i, maj)).tolist()
    
    def _dist_fn():
        """
        Selects the distance function according to the distance paramenter.
        
        Returns
        -------
        A distance function.
        """
        if distance == "EU":
            return _eu
        elif distance == "CH":
            return _ch
        elif distance == "KL":
            return _kl
        elif distance == "HE":
            return _he
        elif distance == "TV":
            return _tv
        elif distance == "CS":
            return _cs
        else:
            raise ValueError("Bad distance function parameter. " + \
                    "Should be one in EU, CH, KL, HE, TV, or CS")
    
    _, class_counts = np.unique(classes, return_counts=True)
    empirical_distribution = class_counts / class_counts.sum()
    K = len(class_counts)
    e = 1 / K
    m = _min_classes(empirical_distribution, e)
    i_m = _i_m(K, m)
    dfn = _dist_fn()
    dist_ed = dfn(empirical_distribution, e)
    return 0.0 if dist_ed == 00 else (dist_ed / dfn(i_m, e)) + (m - 1)
