# Imbalance-degree

A python implementation of the imbalance-degree measure for multi-class 
imbalanced datasets characterization.

This measure is proposed in [[1]](#ref_1) as an alternative for the well known
imbalance-ratio used with binary class imbalanced datasets.

## Authors

This implementation was developed and maintained by [Mario Juez-Gil](mailto:mariojg@ubu.es) from 
[ADMIRABLE](https://www.admirable-ubu.es) research group of the University of 
Burgos, with the help and useful advice from 
[Álvar Arnaiz-González](https://scholar.google.es/citations?user=_9C0tpMAAAAJ&hl=es),
[Juan J. Rodriguez](https://scholar.google.es/citations?user=p4m8t6oAAAAJ&hl=es), 
and his PhD thesis supervisors: 
[César García-Osorio](https://scholar.google.es/citations?user=X08I-_4AAAAJ&hl=es), 
and [Carlos López-Nozal](https://scholar.google.es/citations?user=JAS4N-oAAAAJ&hl=es).

---

## Usage

This module exposes a function called imbalance_degree which takes two 
arguments:

- `classes` : A list of classes (targets) of each instance of the dataset.
- `distance` : distance or similarity function identifier. It can take the 
    following values (`EU` by default):
    - `EU`: Euclidean distance.
    - `CH`: Chebyshev distance.
    - `KL`: Kullback Leibler divergence.
    - `HE`: Hellinger distance.
    - `TV`: Total variation distance.
    - `CS`: Chi-square divergence.

### Example

An usage example could be:

`example.py`

```python
from imbalance_degree import imbalance_degree
import numpy as np

classes = np.array([0,0,0,1,1,2])
print(imbalance_degree(classes, "EU"))
```

`output:`

```
0.49999999999999994
```

---

## References

<a name="ref_1"></a>[1] J. Ortigosa-Hernández, I. Inza, and J. A. Lozano, 
            “Measuring the class-imbalance extent of multi-class problems,” 
            Pattern Recognit. Lett., 2017. DOI: 
            [10.1016/j.patrec.2017.08.002](https://doi.org/10.1016/j.patrec.2017.08.002)

---

## License

Licensed under the [GNU GPLv3](https://opensource.org/licenses/GPL-3.0), 
please see the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

This work was partially supported by the Consejería de Educación of the 
Junta de Castilla y León and by the European Social Fund with the 
EDU/1100/2017 pre-doctoral grants; by the project TIN2015-67534-P 
(MINECO/FEDER, UE) of the Ministerio de Economía Competitividad of the 
Spanish Government and the project BU085P17 (JCyL/FEDER, UE) of the Junta de 
Castilla y León both cofinanced from European Union FEDER funds.