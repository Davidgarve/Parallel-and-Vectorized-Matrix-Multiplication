package org.example;

import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class ParallelMatrixMultiplication {

    private static final int THRESHOLD = 200; // Ajustado para matrices más grandes

    // Clase para transponer matrices
    private static double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // Tarea para realizar multiplicación paralela
    static class MatrixMultiplicationTask extends RecursiveTask<Void> {
        private final double[][] A;
        private final double[][] B;
        private final double[][] C;
        private final int startRow;
        private final int endRow;

        public MatrixMultiplicationTask(double[][] A, double[][] B, double[][] C, int startRow, int endRow) {
            this.A = A;
            this.B = B;
            this.C = C;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        protected Void compute() {
            int numRows = endRow - startRow;

            if (numRows <= THRESHOLD) {
                // Multiplicación secuencial para bloques pequeños
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < B.length; j++) { // Usamos matriz transpuesta
                        double sum = 0;
                        for (int k = 0; k < B[0].length; k++) {
                            sum += A[i][k] * B[j][k];
                        }
                        C[i][j] = sum;
                    }
                }
            } else {
                // Dividir la tarea en dos subtareas más pequeñas
                int mid = startRow + numRows / 2;
                MatrixMultiplicationTask task1 = new MatrixMultiplicationTask(A, B, C, startRow, mid);
                MatrixMultiplicationTask task2 = new MatrixMultiplicationTask(A, B, C, mid, endRow);
                invokeAll(task1, task2);
            }
            return null;
        }
    }

    public static double[][] multiplyMatricesParallel(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsB = B[0].length;

        // Transponer la matriz B para acceso eficiente
        double[][] transposedB = transposeMatrix(B);

        // Inicializar la matriz de resultado
        double[][] C = new double[rowsA][colsB];

        // Usar ForkJoinPool para paralelización
        ForkJoinPool pool = ForkJoinPool.commonPool();
        pool.invoke(new MatrixMultiplicationTask(A, transposedB, C, 0, rowsA));

        return C;
    }
}
