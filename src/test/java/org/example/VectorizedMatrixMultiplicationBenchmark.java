package org.example;

import org.openjdk.jmh.annotations.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(1)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.MILLISECONDS)
public class VectorizedMatrixMultiplicationBenchmark {

    @Param({"100", "500", "1000", "2000", "3000"})
    private int n;

    private List<Long> memoryUsages;

    @State(Scope.Thread)
    public static class Operands {
        public float[][] a;
        public float[][] b;

        @Setup
        public void setup(VectorizedMatrixMultiplicationBenchmark benchmarking) {
            Random random = new Random();
            int size = benchmarking.n;
            a = new float[size][size];
            b = new float[size][size];

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    a[i][j] = random.nextFloat();
                    b[i][j] = random.nextFloat();
                }
            }
        }
    }

    @Setup(Level.Trial)
    public void setupTrial() {
        memoryUsages = new ArrayList<>();
    }

    @Setup(Level.Invocation)
    public void setupInvocation() {
        System.gc();
    }

    @Benchmark
    public void matrixMultiplication(Operands operands) {
        measureMemoryUsage(() -> {
            VectorizedMatrixMultiplication.vectorizedMultiply(operands.a, operands.b);
        });
    }

    private void measureMemoryUsage(Runnable multiplicationTask) {
        Runtime runtime = Runtime.getRuntime();
        long beforeMemory = runtime.totalMemory() - runtime.freeMemory();
        multiplicationTask.run();
        long afterMemory = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = afterMemory - beforeMemory;
        memoryUsages.add(memoryUsed);
    }

    @TearDown(Level.Trial)
    public void tearDownTrial() {
        long totalMemoryUsed = memoryUsages.stream().mapToLong(Long::longValue).sum();
        double averageMemoryUsed = (double) totalMemoryUsed / memoryUsages.size();
        System.out.printf("\nTotal memory used during trial: %.2f MB%n", (double) totalMemoryUsed / (1024 * 1024));
        System.out.printf("Average memory used during trial: %.2f MB%n", averageMemoryUsed / (1024 * 1024));
    }
}