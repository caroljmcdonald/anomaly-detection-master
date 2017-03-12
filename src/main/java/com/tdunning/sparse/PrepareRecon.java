package com.tdunning.sparse;

import com.google.common.io.Resources;
import org.apache.mahout.math.*;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Formatter;

/**
 * Read a bunch of EKG data, chop out windows and cluster the windows. Then
 * reconstruct the signal and figure out the error.
 */
public class PrepareRecon {

    public static void main(String[] args) throws IOException {
        // read the data
        URL x = Resources.getResource("a02.dat");
        double t0 = System.nanoTime() / 1e9;
        Vector trace = Trace.read16b(new File(x.getPath()), 1.0 / 200);
        double t1 = System.nanoTime() / 1e9;
        System.out.printf("Read test data from %s in %.2f s\n", x, t1 - t0);

        final int WINDOW = 32;
        try (Formatter out = new Formatter("toReconstruct.tsv")) {
            for (int i = 0; i + WINDOW < trace.size(); i += WINDOW / 2) {
                // copy chunk of data to temporary window storage and multiply by window
                WeightedVector row = new WeightedVector(new DenseVector(WINDOW), 1, i);
                row.assign(trace.viewPart(i, WINDOW));

                String separator = "";
                for (Vector.Element element : row.all()) {
                    out.format("%s%.4f", separator, element.get());
                    separator = "\t";
                }
                out.format("\n");
            }
        }
        t1 = System.nanoTime() / 1e9;
        System.out.printf("Output in %.2f s\n", t1 - t0);
    }
}
