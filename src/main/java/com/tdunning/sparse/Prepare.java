package com.tdunning.sparse;

import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import org.apache.mahout.clustering.streaming.cluster.BallKMeans;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Formatter;
import java.util.List;

/**
 * Read a bunch of EKG data, chop out windows and cluster the windows. Then
 * reconstruct the signal and figure out the error.
 */
public class Prepare {

    public static void main(String[] args) throws IOException {
        // read the data
        URL x = Resources.getResource("a02.dat");
        double t0 = System.nanoTime() / 1e9;
        Vector trace = Trace.read16b(new File(x.getPath()), 1.0 / 200);
        double t1 = System.nanoTime() / 1e9;
        System.out.printf("Read test data from %s in %.2f s\n", x, t1 - t0);

        final int WINDOW = 32;
        int STEP = 2;
        int SAMPLES = 200000;

        // set up the window vector
        Vector window = new DenseVector(WINDOW);
        for (int i = 0; i < WINDOW; i++) {
            double w = Math.sin(Math.PI * i / (WINDOW - 1.0));
            window.set(i, w * w);
        }

        // window and normalize the data
        t0 = System.nanoTime() / 1e9;
        List<WeightedVector> r = Lists.newArrayList();
        for (int i = 0; i < SAMPLES; i++) {
            int offset = i * STEP;
            WeightedVector row = new WeightedVector(new DenseVector(WINDOW), 1, i);
            row.assign(trace.viewPart(offset, WINDOW));
            row.assign(window, Functions.MULT);
            row.assign(Functions.mult(1 / row.norm(2)));
            r.add(row);
        }
        t1 = System.nanoTime() / 1e9;
        System.out.printf("Windowed data in %.2f s\n", t1 - t0);

        try (Formatter out = new Formatter("prepared.tsv")) {
            for (Vector v : r) {
                String separator = "";
                for (Vector.Element element : v.all()) {
                    out.format("%s%.4f", separator, element.get());
                    separator = "\t";
                }
                out.format("\n");
            }
        }

    }
}
