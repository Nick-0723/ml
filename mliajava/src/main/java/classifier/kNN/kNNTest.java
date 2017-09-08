package classifier.kNN;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static classifier.kNN.kNN.classify0;
import static java.lang.System.out;

public class kNNTest {
    public static Object[] createDataSet() {
        INDArray dataSet = Nd4j.create(new double[]{1.0f, 1.1f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.1f}, new int[]{4, 2});
        String[] labels = new String[]{"A", "A", "B", "B"};
        return new Object[]{dataSet, labels};
    }

    public static void main(String[] args) {
        Object[] obs = createDataSet();
        INDArray inX = Nd4j.create(new double[]{1.0f, 1.1f}, new int[]{1, 2});
        out.printf("classify: %s\n", classify0(inX, (INDArray) obs[0], (String[]) obs[1], 1));
    }

}
