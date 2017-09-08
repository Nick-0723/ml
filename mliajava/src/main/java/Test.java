import sun.awt.image.ImageWatched;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.IntFunction;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Test {
    public static void  main(String[] args){
        List<Integer> list = Arrays.asList(1, 2, 3, 4);
        List<Integer> sum = list.stream()
                .reduce(new LinkedList<Integer>(),
                        (a, b) -> { a.add(b); return a;},
                        (a, b) ->{ a.addAll(b);return a;});
        IntStream s = "123456".chars();
        IntStream  v = s.map(x -> {return x - '0';});
      //  v.forEach(x -> System.out.println(x));

        Stream<List<Integer>> inputStream = Stream.of(
                Arrays.asList(1),
                Arrays.asList(2, 3),
                Arrays.asList(4, 5, 6)
        );
        Stream<Integer> os = inputStream.
                flatMap((childList) -> childList.stream());
        Integer[] ssss = os.toArray(size -> new Integer[size]);
        for(Integer i : ssss){
            System.out.println(i);
        }
    }
}
