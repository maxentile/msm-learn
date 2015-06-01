**Meeting notes: May 28, 2015**

Participants:
- Josh Fass
- John Chodera

**Summary of main points:**
- MSMs:
  - 3 main issues covered: slow projections, kinetic clustering, model interpretation. Also discussed connections to path sampling.
- Experimental component:
  - Goal 1: Become familiar with capabilities of lab automation equipment
  - Goal 2: Do experiments that allow comparisons with MSM predictions (esp. FLiK assays)
  - Goal 3: Continue brainstorming automatable "all-against-all" kinase:drug assays, esp. ideas that can leverage cheap sequencing

**Details...**

MSM construction:
- Projection
  - Goal: efficiently compute maximally metastable nonlinear projections
  - Idea 1: Use existing nonlinear extensions of slow feature analysis (SFA)
    - Justification: the linear tICA solution with time-lag 1 is equivalent to standard SFA (http://www.neuroinformatik.rub.de/PEOPLE/wiskott/Publications/BlasBerkWisk2006-SFAvsICA-NeurComp.pdf). SFA has a few useful features:
     - It can accept multiple time-lagged covariance matrices.
     - Like tICA, it can be trained in "chunks."
     - There are existing nonlinear extensions with fast implementations (xSFA, implemented in MDP): http://mdp-toolkit.sourceforge.net/node_list.html#mdp.nodes.XSFANode
  - Related idea:
    - Explicitly construct nonlinear feature expansion, then do linear tICA in this feature space. In the case of polynomial features, this should provide a baseline for comparison to xSFA. There's a simple library function for this in scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
  - Idea 2: Cheaply guess the main metastable conformations, then construct the smallest subspace that spans those conformations. If we made good guesses, then this might be a good "slow subspace."
    - Tactics for generating metastable state guesses:
      - Use outputs from Ensembler?
      - Use minima of a smoothed potential?
        - Problem: the locations of minima in the smoothed potential may not be near the minima of the original potential
        - Further references on problems with potential-smoothing:
          - David Shalloway
          - Jay Ponder
      - Use conformations sampled by AMD? (accelerated molecular dynamics-- see the custom integrator implemented in OpenMM)
- Clustering
  - Functional goal: compute a clustering of conformations such that each cluster is maximally metastable.
  - Tactic: find optimal weighting of atoms in RMSD or RMSD-similar metric. Intuitively, we want to de-emphasize atoms that fluctuate too quickly.
  - Related work:
    - Simple "RMSD-similar" function: Binet-Cauchy kernel: http://bioinformatics.oxfordjournals.org/content/30/6/784.long
  - Challenges:
    - Formulating a sensible objective to optimize
      - Need to impose some constraints to keep from returning pathological solutions, such as putting all the weight on just one atom.
      - Possible ways to encode the constraint:
        - Require the zero-norm (the number of non-zero entries) of the weight vector to be greater than some number
    - Optimizing the objective:
      - General approaches:
        - formulate a dual optimization problem?
        - Use a geometric approach to imposing the feasibility constraints, e.g. manifold optimization?
- Interpretation: once a MSM is constructed, we would like to extract humanly interpretable insights from it.
  - Visualization tools:
    - Kinetically hierarchical representation (microstates on)
    - Clustering based
    - Reaction coordinate based visualization. Assuming we have a good set of slow coordinates
      - Problem: the inverse transform is usually meaningless, since very small distances in dihedral space can result in very large changes in the overall conformation
      - Possible solutions:
        -
      - Talked with Kyle afterward: Robert McGibbon has already created almost exactly this tool: https://github.com/rmcgibbo/projector
  - Automated natural-language descriptions?
    - There are some sufficiently standardized ways to interpret aspects of a MSM, e.g. the implied timescale plots, that we could probably automate the natural-language description of the results.
    - Related work:
      - The "Automatic statistician" project: http://www.automaticstatistician.com/
  - Construct plots of productive flux based on transition path theory: see Vincent Voelz's work

**Experimental component:**
- FLiK assays: require protein expression-- talk with Patrick about cell-free expression
- PD-seq: http://www.pnas.org/content/110/24/E2153.full
  - Functional goal: identify a large number of functionally relevant proteins based on interactions with a target small molecule, esp. ones that are evolutionarily accessible.
  - Key tactic: find some way to keep each protein and the information that encodes it in the same particle, so that we can cheaply determine which proteins we've selected for in a subsequent batch sequencing step
  - Challenges:
    - Phage display: may be harder to execute than expected, phage contamination is too easy and costly
    - Proteins that don't express well in *E. coli* probably won't be expressed well in this assay-- but it might be possible to normalize afterward using a nonspecific natural product
  - Alternatives:
    - mRNA-display: a more direct way to associate a protein with the nucleic acid that encodes it: http://en.wikipedia.org/wiki/mRNA_display
    - [Rahul Satija](http://www.satijalab.org/publications.html)'s work: nanodroplets for barcoded single-cell transcriptomics (Drop-Seq: http://www.cell.com/cell/abstract/S0092-8674(15)00549-8)
    - Angela Koehler's work on small-molecule arrays: http://www.broadinstitute.org/scientific-community/science/programs/csoft/chemical-biology/group-koehler/chemical-biology-koehler-
  - Other tools/protocols to consider:
    - GAIIX sequencer: these are available very cheaply and have been used in other
    - Wil Greenleaf, ATAC-seq: http://greenleaf.stanford.edu/portfolio_details_buenrostro_2013_nature_methods.html

**Other points:**
- Review of Bayesian forcefield meeting with Theo:
  - Goal: reduce the cost of likelihood evaluations by fitting and evaluating cheaper approximations near/between the points where the full model has been evaluated. [?]
  - Things to look up: [amortized inference](http://web.mit.edu/sjgershm/www/GershmanGoodman14.pdf), [sparse gaussian processes](http://jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf)
- Measuring the metastability of an embedding
  - Question: Is there a way to measure this that doesn't depend on the quality of the subsequent clustering?
    - Answer: probably not, but if we "over-cluster" (and use the same number of clusters in each embedding to be compared), then we should be fine: see the error bounds proved by Marco Sarich
      - Problem: it's difficult to create a differentiable loss function that directly optimizes for metastability.
- Hierarchical MSMs?
  - Motivation: A hierarchical state-space representation would be a natural fit for the structure we expect. It may also provide a route to a more compact representation of the model than the full microstate transition matrix. Also, it may avoid the problem where it's not possible to block-diagonalize the microstate transition matrix due to violation of the assumption that the magnitudes of off-diagonal terms are very small. Also, if two timescales are very close, their corresponding eigenspaces will mix (cf. the 1D potential example).
  - Potential learning algorithms:
    - "Hierarchical HMMs" (HHHM)
      - Basic idea: construct an HMM where each state is also an HHM.
      - Kevin Murphy's thesis describes a linear-time algorithm for learning and inference in the HHHM, exploiting a PGM interpretation of the model for further speed-ups: http://www.cs.ubc.ca/~murphyk/Papers/hhmm_nips01.pdf
- Connections with path sampling
  - TIS can mostly obviate the need for order parameters.
  - Openpathsampling project:
    - Challenges: parallelizing the code
    - Applications to model systems: host-guest systems

**Other references:**
- Christoph Schuette's thesis: http://page.mi.fu-berlin.de/christo2/publications.html
- Marcus Weber's thesis: https://www.zib.de/weber/
- Susanna Roeblitz's thesis: http://www.zib.de/members/susanna.roeblitz

**Short- and long-term deliverables (in increasing order of difficulty):**
- Apply [spectral biclustering](http://scikit-learn.org/stable/modules/biclustering.html) to a "flat" microstate transition matrix to identify coarse-grained structure. If promising, implement and submit a pull request for a wrapper of the sklearn implementation in MSMbuilder.
- Transfer ownership of private repository containing the low-dimensional embedding work to the Chodera Lab Github group.
  - Consolidate testing scripts so that embedding methods can be compared
- Make a list of required components for FLiK assay
- Flesh out possible connections between TIS and active learning.

Other things discussed outside of this meeting:
- Start working with actual kinase data:
  - Get an abl log-in (talk with Sonya and Kyle about this)
