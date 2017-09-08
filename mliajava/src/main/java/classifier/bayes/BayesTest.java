package classifier.bayes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.IntStream;
import static java.lang.System.out;


public class BayesTest {
    static String[][] data() {
        String[] labels = new String[]{"0", "1", "0", "1", "0", "1"};
        String[] docs = new String[]{
                "my dog has flea problems help please",
                "maybe not take him to dog park stupid",
                "my dalmation is so cute i love him",
                "stop posting stupid worthless garbage",
                "mr licks ate my steak how to stop him",
                "quit buying worthless dog food stupid"};
        return new String[][]{docs, labels};
    }

    public static ArrayList<String> createVocList(String[] docs) {
        Set<String> res = Arrays.stream(docs).reduce(
                new TreeSet<String>(),

                (list, doc) -> {
                    Arrays.stream(doc.split(" ")).reduce(
                            list,
                            (p1, w) -> {
                                p1.add(w.trim().toLowerCase());
                                return p1;
                            },
                            (left, right) -> {
                                left.addAll(right);
                                return list;
                            });
                    return list;
                },

                (left, right) -> {
                    left.addAll(right);

                    return left;
                });
        return new ArrayList<>(res);
    }

    public static INDArray setOfWordsToVec(ArrayList<String> words, String inputStr){
        INDArray vec = Nd4j.zeros(words.size());
        Arrays.stream(inputStr.split(" ")).forEach(x -> vec.put(0, words.indexOf(x.trim().toLowerCase()), 1));
        return vec;
    }
    public static void main(String[] args){
        String[][] datas = data();
        ArrayList<String> words = createVocList(datas[0]);
        words.stream().forEach(x -> System.out.println(x));
        INDArray trainingData = IntStream.range(0, datas[0].length).boxed().reduce(
                Nd4j.zeros(datas[0].length, words.size()),
                (dataSet, idx) ->{
                    dataSet.putRow(idx, setOfWordsToVec(words, datas[0][idx]));
                    return dataSet;

                },
                (left, right) ->{
                    INDArray res = Nd4j.zeros(datas[0].length, words.size());
                    for(int i = 0; i < left.rows(); i++){
                        res.putRow(i, left.getRow(i));
                    }
                    for (int i = 0; i < right.rows(); i++){
                        res.putRow(i + left.rows(), right.getRow(i));
                    }
                    return res;
                }
        );
        INDArray trainingLabel = IntStream.range(0, datas[1].length).boxed().reduce(
                Nd4j.zeros(datas[1].length),
                (dataSet, idx) ->{
                    dataSet.put(0, idx, Double.parseDouble(datas[1][idx]));
                    return dataSet;

                },
                (left, right) ->{
                    INDArray res = Nd4j.zeros(datas[1].length);
                    for(int i = 0; i < left.rows(); i++){
                        res.put(0, i, left.getRow(i));
                    }
                    for (int i = 0; i < right.rows(); i++){
                        res.put(0, i + left.rows(), right.getRow(i));
                    }
                    return res;
                }
        );
        out.println();

        out.println(trainingData);
        out.println();

        out.println(trainingLabel);

        NativeBayes.BayesModel model = NativeBayes.train(trainingData, trainingLabel);
        for(int i = 0; i < trainingData.rows(); i++){
            int res = model.classify(trainingData.getRow(i));
            out.printf("actual class %d, predict class %d\n", trainingLabel.getInt(0, i), res);
        }
        out.println(model.fProb.probability().getDouble(0, words.indexOf("stupid")));
        out.println(model.sProb.probability().getDouble(0, words.indexOf("stupid")));


    }
}
