package classifier.kNN;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

 import static java.lang.System.out;

public class HandWriteringTest {
    static class Data {
        Data(int rows) {
            dataSet = Nd4j.create(rows == 0?1:rows, 1024);
            labels = new String[rows];
        }
        Data() {
            dataSet = Nd4j.create(1, 1024);
            labels = new String[1];
        }
        public INDArray dataSet;
        public String[] labels;
        public int counter;
        public Data merge(Data that){

            int total = this.counter + that.counter;
            Data tData = new Data(total);
            for(int i = 0; i<counter; i++){
                tData.dataSet.putRow(i, this.dataSet.getRow(i));
            }
            tData.counter = tData.dataSet.rows();//counter
            for(int i = tData.counter; i < total; i++)
                tData.dataSet.putRow(i, that.dataSet.getRow(i - this.counter));
            tData.counter = total;
            return tData;
        }
        public Data addRow(String sym, INDArray data){

            Data tData = new Data(counter);
            tData.dataSet = dataSet.dup();
            tData.dataSet = tData.dataSet.repmat(counter+1, 1024);
            tData.dataSet.putRow(counter, data);
            tData.labels = new String[counter + 1];
            System.arraycopy(labels, 0, tData.labels, 0, counter);
            tData.labels[counter] = sym;
            tData.counter++;
            return tData;
        }
    }

    public static Optional<INDArray> image_to_vector(Path fileName) {
        try {
            double[] res = Files.lines(fileName).flatMap(x ->
                    x.chars().map(e -> e - '0').boxed()).mapToDouble(Integer::doubleValue).toArray();
            INDArray r = Nd4j.create(res, new int[]{1, 1024});
            return Optional.of(Nd4j.create(res, new int[]{1, 1024}));
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    public static Optional<INDArray> image_to_vector(String fileName) {
        return image_to_vector(new File(fileName).toPath());
    }

    public static Stream<Path> listDir(String dir)  {
        return Arrays.stream(new File(dir).list()).map(x -> new File(x).toPath());
    }



    public static Data readData(String dataDir) {
        int fileCount = new File(dataDir).list().length;

        return listDir(dataDir).reduce(
                new Data(fileCount),
                (Data data, Path file) -> {
                    INDArray d =  image_to_vector(new File(dataDir, file.toString()).toPath()).get();
                    String sym = file.toString().split("\\.")[0].split("_")[0];
                    data.dataSet.putRow(data.counter, d);
                    data.labels[data.counter] = sym;
                    data.counter++;
                    return data;
                },
                (left, right) -> {
                     return left.merge(right);
                });
    }


    public static void main(String[] args)  {

        String dataDir = "digits/trainingDigits";
        Data trainingData = readData(dataDir);

        String testDir = "digits/testDigits";
        Data testData = readData(testDir);

        INDArray vec = testData.dataSet;

        int errorCount = IntStream.range(0, testData.counter).reduce(
                0,
                (x, y) -> {
                    String label = kNN.classify0(vec.getRow(y), trainingData.dataSet, trainingData.labels, 3);
                    if (!label.equals(testData.labels[y])) {
                        out.printf("the classifier came back with: %s, the real answer is %s\n", label, testData.labels[y]);
                        return x + 1;
                    } else return x;
                });
        out.printf("the total number of samples is: %d\n", testData.counter);
        out.printf("the total number of errors is: %d\n", errorCount);
        out.printf("the total error rate is: %f\n", 1.0*errorCount / testData.counter);
    }

}