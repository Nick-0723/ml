package classifier.bayes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class EmailTest {
    public static Stream<String> parse(String text){
        Pattern p = Pattern.compile("\\W");
        return Arrays.stream(text.split(p.pattern())).filter(x -> x.trim().length() >= 2).map(x -> x.toLowerCase());
    }
    public static ArrayList<String> createVocList(String[] docs) {
        Set<String> res = Arrays.stream(docs).reduce(new TreeSet<String>(),
                (list, doc) -> {
                    parse(doc).reduce(
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
        parse(inputStr).forEach(x -> vec.put(0, words.indexOf(x.trim().toLowerCase()), 1));
        return vec;
    }

}
