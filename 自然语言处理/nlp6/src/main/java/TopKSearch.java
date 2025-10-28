import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class TopKSearch {

    private static final String STOP_WORDS_FILE = "cn_stopwords.txt";  // 停用词文件路径
    private static final String ARTICLE_PATH = "./article/";

    public static void main(String[] args) throws IOException {
        // 读取停用词
        List<String> stopWords = Files.readAllLines(Paths.get(STOP_WORDS_FILE));

        // 预处理
        Map<String, List<String>> documents = readAndProcessDocuments(stopWords);

        // 计算词项频率TF
        Map<String, Map<String, Integer>> termFrequency = calculateTF(documents);

        // 构建胜者表
        Map<String, List<String>> championList = buildChampionList(termFrequency, 5);  // 假设 r = 5

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("请输入查询词（用空格分隔），或者输入 'exit' 退出：");
            String input = scanner.nextLine();
            if (input.equalsIgnoreCase("exit")) {
                break;
            }

            List<String> query = Arrays.asList(input.split("\\s+"));

            System.out.println("请输入K值：");
            int K = scanner.nextInt();
            scanner.nextLine();  // 清除缓冲区

            // 查询Top-K文档
            List<String> topKDocuments = getTopKDocuments(query, K, championList);
            System.out.println("Top K documents: " + topKDocuments);
        }

        scanner.close();
    }

    private static Map<String, List<String>> readAndProcessDocuments(List<String> stopWords) throws IOException {
        Map<String, List<String>> documents = new HashMap<>();
        Segment segment = HanLP.newSegment();

        for (int i = 1; i <= 10000; i++) {
            String filePath = ARTICLE_PATH + i + ".txt";
            String content = new String(Files.readAllBytes(Paths.get(filePath)));
            List<String> tokens = segment(content, segment);
            tokens = tokens.stream()
                    .filter(token -> !stopWords.contains(token))
                    .collect(Collectors.toList());
            documents.put(String.valueOf(i), tokens);
        }
        return documents;
    }

    private static List<String> segment(String text, Segment segment) {
        List<Term> termList = segment.seg(text);
        return termList.stream()
                .map(term -> term.word)
                .collect(Collectors.toList());
    }

    private static Map<String, Map<String, Integer>> calculateTF(Map<String, List<String>> documents) {
        Map<String, Map<String, Integer>> termFrequency = new HashMap<>();

        for (String docId : documents.keySet()) {
            List<String> tokens = documents.get(docId);
            Map<String, Integer> tf = new HashMap<>();
            for (String token : tokens) {
                tf.put(token, tf.getOrDefault(token, 0) + 1);
            }
            termFrequency.put(docId, tf);
        }

        return termFrequency;
    }

    private static Map<String, List<String>> buildChampionList(Map<String, Map<String, Integer>> termFrequency, int r) {
        Map<String, List<String>> championList = new HashMap<>();
        Map<String, PriorityQueue<Map.Entry<String, Integer>>> maxHeaps = new HashMap<>();

        for (String docId : termFrequency.keySet()) {
            for (Map.Entry<String, Integer> entry : termFrequency.get(docId).entrySet()) {
                String term = entry.getKey();
                int frequency = entry.getValue();

                maxHeaps.putIfAbsent(term, new PriorityQueue<>(Comparator.comparingInt(Map.Entry::getValue)));
                PriorityQueue<Map.Entry<String, Integer>> heap = maxHeaps.get(term);

                heap.offer(new AbstractMap.SimpleEntry<>(docId, frequency));
                if (heap.size() > r) {
                    heap.poll();
                }
            }
        }

        for (Map.Entry<String, PriorityQueue<Map.Entry<String, Integer>>> entry : maxHeaps.entrySet()) {
            List<String> topDocs = entry.getValue().stream()
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());
            Collections.reverse(topDocs);
            championList.put(entry.getKey(), topDocs);
        }

        return championList;
    }

    private static List<String> getTopKDocuments(List<String> query, int K, Map<String, List<String>> championList) {
        Map<String, Integer> docFrequency = new HashMap<>();

        for (String term : query) {
            List<String> topDocs = championList.get(term);
            if (topDocs != null) {
                for (String docId : topDocs) {
                    docFrequency.put(docId, docFrequency.getOrDefault(docId, 0) + 1);
                }
            }
        }

        List<String> intersectDocs = docFrequency.entrySet().stream()
                .filter(entry -> entry.getValue() == query.size())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        Map<String, Double> docScores = new HashMap<>();
        for (String term : query) {
            List<String> topDocs = championList.get(term);
            if (topDocs != null) {
                for (String docId : topDocs) {
                    if (intersectDocs.contains(docId)) {
                        docScores.put(docId, docScores.getOrDefault(docId, 0.0) + 1);
                    }
                }
            }
        }

        return docScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(K)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
}
