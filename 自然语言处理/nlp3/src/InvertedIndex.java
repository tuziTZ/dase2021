import java.io.*;
import java.util.*;

public class InvertedIndex {
    //词的最大长度
    private static final int MAX_LENGTH = 10;

    private Map<String, Set<String>> invertedIndex = new HashMap<>();

    public void buildIndex(String directoryPath, List<String> dict,List<String> stopwords) {
        File directory = new File(directoryPath);
        if (!directory.isDirectory()) {
            System.out.println("Invalid directory path.");
            return;
        }
        //遍历所有文档
        for (File file : directory.listFiles()) {
            if (file.isFile() && file.getName().endsWith(".txt")) {
                //读取文件
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        //调用分词算法
                        List<String> words = segment(line, dict);
                        for (String word : words) {
                            if (!stopwords.contains(word)) {

                                //加入哈希链表
                                invertedIndex.computeIfAbsent(word, k -> new HashSet<>()).add(file.getName());
                            }
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    //最大正向匹配分词算法
    public static List<String> segment(String text, List<String> dict) {
        List<String> result = new ArrayList<>();
        int textLength = text.length();
        int currentIndex = 0;

        while (currentIndex < textLength) {
            int maxLength = Math.min(MAX_LENGTH, textLength - currentIndex);
            boolean found = false;

            for (int length = maxLength; length > 0; length--) {
                String subStr = text.substring(currentIndex, currentIndex + length);
                //如果在词表找到对应的词，就从下一个字开始
                if (dict.contains(subStr)) {
                    result.add(subStr);
                    currentIndex += length;
                    found = true;
                    break;
                }
            }
            //如果没有找到就把单字切割出来
            if (!found) {
                result.add(Character.toString(text.charAt(currentIndex)));
                currentIndex++;
            }
        }
        //包含分词的列表
        return result;

    }
    public Set<String> search(String query) {
        long startTime = System.nanoTime();

        Set<String> result = invertedIndex.getOrDefault(query, new HashSet<>());

        long endTime = System.nanoTime();
        long elapsedTime = endTime - startTime;
        System.out.println("Search time: " + elapsedTime + " nanoseconds");

        return result;
    }

    public static void main(String[] args) {
        List<String> dict = new ArrayList<>();
        List<String> stopwords = new ArrayList<>();
        try {
            BufferedReader readerDict = new BufferedReader(new FileReader("corpus.dict.txt"));
            String line;
            while ((line = readerDict.readLine()) != null) {
                dict.add(line.trim());
            }
            readerDict.close();

            BufferedReader readerStopwords = new BufferedReader(new FileReader("cn_stopwords.txt"));
            while ((line = readerStopwords.readLine()) != null) {
                stopwords.add(line.trim());
            }
            readerStopwords.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        InvertedIndex index = new InvertedIndex();
        index.buildIndex("./article", dict,stopwords);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("请输入查询词汇：");
            String query = scanner.nextLine().trim();
            if ("exit".equalsIgnoreCase(query)) {
                break;
            }

            Set<String> result = index.search(query);
            if (result.isEmpty()) {
                System.out.println("无匹配文档");
            } else {
                System.out.println("该词汇出现在 "+result.size()+" 个文档中，分别是：" + result);
            }
        }

        scanner.close();
    }
}