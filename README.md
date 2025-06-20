# Purpose

Performs global sequence alignment of two strings.  Allows for affine gap penalties.

# Installation

The package can be installed via `pip`, `conda`, or `mamba`.

## pip

Run the following in a terminal.

```
python3 -m venv my_venv_for_globalign
source my_venv_for_globalign/bin/activate
pip install globalign
```

# Contributing

You will need to install [quarto](https://quarto.org/) and [quartodoc](https://github.com/machow/quartodoc/) to be able to publish documentation updates.  To build the documentation locally and publish it to GitHub, do the following: From the project root, run:

```bash
quartodoc build
quarto render
```

If you are happy with the resulting website, then push your changes to a branch other than `gh-pages`.  Then, (from a branch other than `gh-pages`), execute the command:

```bash
quarto publish gh-pages
```

The public-facing website should now be updated.

To run unit tests, make sure [pytest](https://docs.pytest.org/en/stable/) and [hatch](https://hatch.pypa.io/latest/) are installed and available.  From the project root, run

```bash
rm -rf dist/
hatch build
hatch test
```

To install from source, run the following from the project root after building:

```bash
pip install --editable .
```

Versions can be changed via git tags.  For example, run this with the version you want:

```bash
git tag -a "v0.0.0"
```

# Acknowledgements

A special thanks goes to Mykola Akulov and Ragnar Groot Koerkamp for their insightful [blog post](https://curiouscoding.nl/posts/alignment-scores-transform/) without which I would not have known how to make this package work with both scoring and costing schemes.

# References

1. https://web.stanford.edu/class/cs262/archives/presentations/lecture3.pdf
2. https://ocw.mit.edu/courses/6-096-algorithms-for-computational-biology-spring-2005/01f55f348ea1e95f7015bd1b40586012_lecture5.pdf
3. Martin Mann, Mostafa M Mohamed, Syed M Ali, and Rolf Backofen
     Interactive implementations of thermodynamics-based RNA structure and RNA-RNA interaction prediction approaches for example-driven teaching
     PLOS Computational Biology, 14 (8), e1006341, 2018.
4. Martin Raden, Syed M Ali, Omer S Alkhnbashi, Anke Busch, Fabrizio Costa, Jason A Davis, Florian Eggenhofer, Rick Gelhausen, Jens Georg, Steffen Heyne, Michael Hiller, Kousik Kundu, Robert Kleinkauf, Steffen C Lott, Mostafa M Mohamed, Alexander Mattheis, Milad Miladi, Andreas S Richter, Sebastian Will, Joachim Wolff, Patrick R Wright, and Rolf Backofen
     Freiburg RNA tools: a central online resource for RNA-focused research and teaching
     Nucleic Acids Research, 46(W1), W25-W29, 2018.
5. An improved algorithm for matching biological sequences. Osamu Gotoh. https://doi.org/10.1016/0022-2836(82)90398-9
6. http://www.cs.cmu.edu/~durand/03-711/2017/Lectures/Sequence-Alignment-2017.pdf
7. https://bioboot.github.io/bimm143_W20/class-material/nw/
8. https://www.ncbi.nlm.nih.gov/CBBresearch/Przytycka/download/lectures/PCB_Lect02_Pairwise_allign.pdf
9. https://ics.uci.edu/~xhx/courses/CS284A-F08/lectures/alignment.pdf
10. https://link.springer.com/chapter/10.1007/978-3-319-90684-3_2
11. Optimal sequence alignment using affine gap costs. https://link.springer.com/content/pdf/10.1007/BF02462326.pdf
12. Optimal alignments in linear space. Eugene W. Myers, Webb Miller.  https://doi.org/10.1093/bioinformatics/4.1.11
13. Sequence alignment using FastLSA. https://webdocs.cs.ualberta.ca/~duane/publications/pdf/2000metmbs.pdf
14. MASA: A Multiplatform Architecture for Sequence Aligners
        with Block Pruning. https://doi.org/10.1145/2858656
15. https://community.gep.wustl.edu/repository/course_materials_WU/annotation/Introduction_Dynamic_Programming.pdf
16. Optimal gap-affine alignment in O(s) space. https://doi.org/10.1093/bioinformatics/btad074
17. Exact global alignment using A* with chaining seed heuristic and match pruning.
    https://doi.org/10.1093/bioinformatics/btae032
18. Transforming match bonus into cost. https://curiouscoding.nl/posts/alignment-scores-transform/
19. Improving the time and space complexity of the WFA algorithm and generalizing its scoring.
        https://doi.org/10.1101/2022.01.12.476087
20. A* PA2: up to 20 times faster exact global alignment.
        https://doi.org/10.1101/2024.03.24.586481
21. Notes on Dynamic-Programming Sequence Alignment.
        https://globin.bx.psu.edu/courses/fall2001/DP.pdf
22. Lecture 6: Affine gap penalty function.
        https://www.cs.hunter.cuny.edu/~saad/courses/compbio/lectures/lecture6.pdf