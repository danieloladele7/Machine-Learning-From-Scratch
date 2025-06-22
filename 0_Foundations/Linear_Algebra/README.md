# ROADMAP

## Module 1: Fundamental Concepts in Linear Algebra and RoadMap 2024

This module reinforces essential pre-requisites like real numbers, vector norms, and geometric basics using the Cartesian coordinate system, setting a strong foundation for advanced linear algebra applications as well as Linear Algebra RoadMap for 2024.

- Real Number & Vector SPace.
- Norm of Vector
- Cartesian Co-ordinate System
- Angles and Triogonmetry
- Norm vs Euclidean Distance
- Pythangorean Theorem & Orthogonality

## Module 2: Vector Spaces and Operations

This module delves deep into vector operations, covering fundamental concepts, special vectors, and operations such as dot products and the Cauchy-Schwarz inequality, Linear Combination, Geometric Intuition and Interpretation emphasizing their applications in machine learning and analytical geometry.

- Foundations of Vectors
- Special Vectors and Operations
- Advanced Vector Concepts
- Dot Product and Applications
- Cauchy-Schwarz

## Module 3&4: Matrices and Solving Linear Systems

This module introduces the foundational concepts of matrices, including their types and structures, and delves into basic matrix operations such as addition, subtraction, and scalar multiplication, as well as Matrix Multiplications essential for performing more complex algebraic manipulations. It also covers advanced techniques like Gaussian elimination, REF, RREF (reduces row echelon forms) and the applications of matrices in solving linear systems, null space, column space, basis and ranks, emphasizing critical thinking and problem-solving skills in algebra.

- Foundations of Linear Systems & Matrices
- Introduction to Matrices
- Core Matrix Operations
- Gaussian Reduction REF, RREF
- Null Space Column Space, Rank, Full Rank.

## Module 5: Matrix Laws and Linear Transformations

Dives Deep into the concepts of matrix determinant, inverses and transpose operations, detailing methods for calculating inverses of 2x2 and 3x3 matrices and their practical applications, while also explaining the significance of these transformations in relation to the dot product and matrix functionality.

- Algebriac Laws for Matrices with Proofs.
- Determinants and Their Properties.
- Transpose and Inverses of Matrices.
- Transpose of Matrices

## Module 6: Advanced Linear Algebra - From Projections to Matrix Factorization

This unit delves into sophisticated concepts in vector spaces and matrix operations, including projections, the Gram-Schmidt process, special matrix properties, matrix factorizations, and decompositions like QR and SVD, equipping students with advanced tools for complex problem-solving in data science and AI.

- Vector Spaces and Projections.
- Gram-Schmidt Process.
- Matrix Factorisation
- QR Decomposition
- Eigenvalues, Eigenvectors, and Eigen Decomposition.
- Singular Value Decomposition (SVD)

## Resource

- Fundamentals to Linear Algebra by [ LUNARTECH ](https://academy.lunartech.ai/product/fundamentals-to-linear-algebra) and
- Linear Algebra Course – Mathematics for Machine Learning and Generative AI by [FreeCodeCamp](https://www.youtube.com/watch?v=rSjt1E9WHaQ)
- Mathematics for Computer [Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/)
- The Modern Mathematics of Deep [Learning](https://arxiv.org/pdf/2105.04026)
- Geometric Deep Learning Grids, Groups, Graphs, Geodesics, and [Gauges](https://arxiv.org/pdf/2104.13478v2)
- The Principles of Deep Learning [Theory](https://arxiv.org/pdf/2106.10165)

## Important

- No Code Changes + CUML equals 50x Speedup for Sklearn [video](https://www.youtube.com/watch?v=lWdazhubMrc)
- Install [tf](https://www.tensorflow.org/install/pip), [Rapid](https://docs.rapids.ai/install/) use rapid with [colab](https://colab.google/articles/cudf)

  ```shell
  python3 -m pip install 'tensorflow[and-cuda]'
  # Verify the installation:
  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

  # Install PyTorch
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

  # other packages
  pip3 install pandas matplotlib scikit-learn tensorboard xgboost

  pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.6.*" "dask-cudf-cu12==25.6.*" "cuml-cu12==25.6.*" \
    "cugraph-cu12==25.6.*" "nx-cugraph-cu12==25.6.*" "cuxfilter-cu12==25.6.*" \
    "cucim-cu12==25.6.*" "pylibraft-cu12==25.6.*" "raft-dask-cu12==25.6.*" \
    "cuvs-cu12==25.6.*" "nx-cugraph-cu12==25.6.*"
  ```
