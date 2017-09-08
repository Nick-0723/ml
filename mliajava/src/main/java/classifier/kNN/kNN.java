package classifier.kNN;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class kNN {

    public static String classify0(INDArray inX, INDArray dataSet, String[] labels, int k) {
        INDArray distances = Nd4j.create(dataSet.rows(), 1);
        for (int i = 0; i < dataSet.shape()[0]; i++) {
            distances.put(i, 0, dataSet.getRow(i).distance2(inX));
        }
        INDArray[] sortedDistIndicies = Nd4j.sortWithIndices(distances, 0, true);
        INDArray indecs = sortedDistIndicies[0].get(NDArrayIndex.interval(0, k), NDArrayIndex.all());

        Stream<Integer> kIndexs = IntStream.of(NDArrayUtil.toInts(indecs)).boxed();

        Map<String, Integer> counter = kIndexs.reduce(
                new TreeMap<String, Integer>(),
                (c, index) -> {
                    String vote = labels[index];
                    c.put(vote, c.getOrDefault(vote, new Integer(0)) + 1);
                    return c;
                },
                (left, right) -> {
                    left.putAll(right);
                    return left;
                });

        Set<Map.Entry<String, Integer>> rc = new TreeSet<Map.Entry<String, Integer>>((x1, x2) -> {
            return x2.getValue() - x1.getValue() != 0 ? x2.getValue() - x1.getValue() : x2.getKey().compareTo(x1.getKey());
         });

        rc.addAll(counter.entrySet());

        return rc.iterator().next().getKey();
    }

    public static INDArray autoNorm(INDArray dataSet) {

        INDArray minVals = Nd4j.create(1, dataSet.columns());
        for (int i = 0; i < dataSet.columns(); i++) minVals = Transforms.min(dataSet.getColumn(i), 1);
        INDArray maxVals = Nd4j.create(1, dataSet.columns());
        for (int i = 0; i < dataSet.columns(); i++) maxVals = Transforms.max(dataSet.getColumn(i), 1);

        INDArray ranges = maxVals.sub(minVals);

        INDArray res = Nd4j.create(dataSet.rows(), dataSet.columns());
        for (int i = 0; i < dataSet.rows(); i++)
            res.putRow(i, dataSet.getRow(i).sub(minVals).div(ranges));

        INDArray retV = Nd4j.create(3, 1);
        retV.putRow(0, res);
        retV.putRow(1, ranges);
        retV.putRow(2, minVals);
        return retV;

    }
}
