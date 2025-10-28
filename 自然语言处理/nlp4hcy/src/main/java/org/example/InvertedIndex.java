package org.example;

import java.io.*;
import java.util.*;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;

public class InvertedIndex {
    private Map<String, Set<Integer>> invertedIndex;

    public InvertedIndex() {
        invertedIndex = new HashMap<>();
    }

    // 读取文件并构建倒排索引
    public void buildIndex(String directory) {
        for (int i = 1; i <= 20738; i++) { // 修改文件数为20738
            String fileName = directory + "/" + i + ".txt";
            try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
                String line;
                while ((line = br.readLine()) != null) { //逐行读取文档内容，直到文件结束。
                    List<Term> terms = HanLP.segment(line);//分词
                    for (Term term : terms) {
                        String word = term.word;//获取当前词语的文本内容
                        if (word.length() > 1) {//过滤掉单字
                            invertedIndex.computeIfAbsent(word, k -> new HashSet<>()).add(i);
                        }//将当前词语添加到倒排索引中。
                        // 如果该词语在倒排索引中不存在，则创建一个新HashSet来存储对应的文件编号，然后将该文件编号添加到 HashSet 中
                        // 如果该词语在倒排索引中已存在，则直接将文件编号添加到对应的HashSet中。
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // 检索词并输出对应的文件ID
//    public Set<Integer> search(String query) {
//        return invertedIndex.getOrDefault(query.toLowerCase(), Collections.emptySet());
//    }

    // 获取与指定词语语义相似度最高的Top2个词
//    public List<String> getTopSimilarWords(String query) {
//        try {
//            Process p = Runtime.getRuntime().exec("python script.py");
//            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
//            String line;
//            while ((line = in.readLine()) != null) {
//                System.out.println(line);
//            }
//            in.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        return Arrays.asList("similar_word1", "similar_word2");
//    }
    public List<String> getTopSimilarWords(String query) {
        List<String> results = new ArrayList<>();
        try {
            Process p = Runtime.getRuntime().exec("python script.py " + query);
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);  // 打印 Python 脚本的输出
                results.add(line);  // 假设每行输出是一个相似的词
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return results;
    }

    // 获取与指定词语语义相似度最高的Top2个词及其对应的文件ID
    public Map<String, Set<Integer>> searchWithSimilarity(String query) {
        Map<String, Set<Integer>> resultMap = new HashMap<>();
        List<String> similarWords = getTopSimilarWords(query);
        for (String word : similarWords) {
            Set<Integer> documentIds = invertedIndex.getOrDefault(word.toLowerCase(), Collections.emptySet());
            resultMap.put(word, documentIds);
        }
        return resultMap;
    }

    public static void main(String[] args) {
        InvertedIndex invertedIndex = new InvertedIndex();
        invertedIndex.buildIndex("article");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Enter search query (or 'exit' to quit): ");
            String query = scanner.nextLine();
            if (query.equalsIgnoreCase("exit")) {
                break;
            }
            long startTime = System.nanoTime();
            Map<String, Set<Integer>> result = invertedIndex.searchWithSimilarity(query);
            long endTime = System.nanoTime();
            long elapsedTime = (endTime - startTime) / 1000000; // Convert to milliseconds
            System.out.println("Search results for '" + query + "': " + result);
            System.out.println("Search time: " + elapsedTime + " milliseconds");
        }
    }
}
