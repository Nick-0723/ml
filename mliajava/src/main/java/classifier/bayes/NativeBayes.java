package classifier.bayes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.stream.IntStream;

import static java.lang.Math.log;

public class NativeBayes {
    static class Prob {
        private INDArray num;
        private Double denom;
        private int rows;

        Prob(INDArray num, Double denom) {
            this.num = num;
            this.denom = denom;
        }

        public INDArray probability() {
            INDArray temp = num.div(denom);
            return temp;
        }

        public INDArray logProbability() {
            return Transforms.log(probability());
        }

        public INDArray merge(INDArray that) {
            if (rows == 0) {

            }
            return num;
        }

        public static Prob apply(int size) {
            return new Prob(Nd4j.zeros(size), 2.0);
        }
    }

    static class BayesModel {
          Prob fProb;
          Prob sProb;
          double fClass;

        public BayesModel(Prob fProb, Prob sProb, double fClass) {
            this.fProb = fProb;
            this.sProb = sProb;
            this.fClass = fClass;
        }

        public int classify(INDArray vec) {
            double p1 = vec.mul(fProb.num).sum(1).getDouble(0, 0) + log(fClass);
            double p2 = vec.mul(sProb.num).sum(1).getDouble(0, 0) + log(1.0 - fClass);
            return p1 > p2 ? 0 : 1;
        }
    }

    public static BayesModel train(INDArray dataSet, INDArray category) {
        int rows = dataSet.rows();
        int cols = dataSet.columns();

        Prob[] probs = IntStream.range(0, rows).boxed().reduce(
                new Prob[]{Prob.apply(cols), Prob.apply(cols)},
                (p1, rowIdx) -> {

                    INDArray v = dataSet.getRow(rowIdx);
                    if (category.getInt(0, rowIdx) == 1) {
                        INDArray n = p1[0].num.add(v);
                        double sum = p1[0].denom + v.sum(1).getDouble(0, 0);
                        return new Prob[]{new Prob(n, sum), p1[1]};
                    } else {
                        INDArray n = p1[1].num.add(v);
                        double sum = p1[1].denom + v.sum(1).getDouble(0, 0);
                        return new Prob[]{p1[0], new Prob(n, sum)};
                    }
                },
                (p1, p2) -> {
                    Prob r1 = Arrays.stream(p1).reduce(
                            Prob.apply(cols),
                            (r, p) -> {
                                r.denom += p.denom;
                                r.num.add(p.num);
                                return r;
                            });
                    Prob r2 = Arrays.stream(p2).reduce(
                            Prob.apply(cols),
                            (r, p) -> {
                                r.denom += p.denom;
                                r.num.add(p.num);
                                return r;
                            });
                    return new Prob[]{r1, r2};
                });
        return new BayesModel(probs[1], probs[0], category.sum(1).getDouble(0, 0) / rows);
    }


}
