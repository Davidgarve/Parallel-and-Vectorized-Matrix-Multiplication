package org.example;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class VectorizedMatrixMultiplication {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static float[][] vectorizedMultiply(float[][] A, float[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;

        float[][] result = new float[rowsA][colsB];

        // Preprocess columns of B
        float[][] transposedB = new float[colsB][colsA];
        for (int i = 0; i < colsA; i++) {
            for (int j = 0; j < colsB; j++) {
                transposedB[j][i] = B[i][j];
            }
        }

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                // Initialize the result cell
                float sum = 0.0f;
                int k = 0;

                // Vectorized computation for chunks of size SPECIES.length()
                for (; k <= colsA - SPECIES.length(); k += SPECIES.length()) {
                    // Load vectors
                    FloatVector aVector = FloatVector.fromArray(SPECIES, A[i], k);
                    FloatVector bVector = FloatVector.fromArray(SPECIES, transposedB[j], k);

                    // Compute dot product
                    sum += aVector.mul(bVector).reduceLanes(VectorOperators.ADD);
                }

                // Handle remaining elements (tail)
                for (; k < colsA; k++) {
                    sum += A[i][k] * transposedB[j][k];
                }

                result[i][j] = sum;
            }
        }
        return result;
    }

}
