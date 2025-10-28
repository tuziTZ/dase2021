import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class HighFrequencyWords {
    //词的最大长度
    private static final int MAX_LENGTH = 7;
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

    public static void main(String[] args) {
        List<String> dict = new ArrayList<>();
        List<String> stopwords = new ArrayList<>();
        //记录词频的map
        Map<String, Integer> wordFrequency = new HashMap<>();
        //记录总词数
        int totalCount = 0;

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

        try {
            BufferedReader readerText = new BufferedReader(new FileReader("corpus.sentence.txt"));
            String line;
            while ((line = readerText.readLine()) != null) {
                List<String> segmentedText = segment(line, dict);
                for (String word : segmentedText) {
                    if (!stopwords.contains(word)) {
                        wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
                        totalCount += 1;
                    }
                }
            }
            readerText.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<Map.Entry<String, Integer>> sortedWordFrequency = new ArrayList<>(wordFrequency.entrySet());
        sortedWordFrequency.sort((entry1, entry2) -> entry2.getValue().compareTo(entry1.getValue()));

        try {
            FileWriter writer = new FileWriter("high_frequency.txt");
            for (int i = 0; i < Math.min(20, sortedWordFrequency.size()); i++) {
                writer.write(sortedWordFrequency.get(i).getKey()+":"+ (double) sortedWordFrequency.get(i).getValue()/totalCount+ "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
