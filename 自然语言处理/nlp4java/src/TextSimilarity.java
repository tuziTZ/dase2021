import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.text.similarity.CosineSimilarity;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.packed.PackedInts.Reader;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import org.apache.commons.lang3.ArrayUtils;

public class TextSimilarity {
    private static final Logger LOGGER = LoggerFactory.getLogger(TextSimilarity.class);
    private static final String BERT_MODEL = "google-bert/bert-base-chinese";

    private static Set<String> stopwords;
    private static Set<String> dictionary;
    private static TokenizerFactory tokenizerFactory;
    private static CosineSimilarity cosineSimilarity;

    static {
        initTokenizerFactory();
        cosineSimilarity = new CosineSimilarity();
        stopwords = loadStopwords("cn_stopwords.txt");
        dictionary = loadDictionary("corpus.dict.txt");
    }

    private static void initTokenizerFactory() {
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    public static void main(String[] args) {
        String documentsDirectory = "./article";
        String stopwordsFile = "cn_stopwords.txt";
        processTextSimilarity(documentsDirectory, stopwordsFile);
    }

    private static Set<String> loadStopwords(String filePath) {
        Set<String> stopwords = new HashSet<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stopwords.add(line.trim());
            }
        } catch (IOException e) {
            LOGGER.error("Error loading stopwords from file: {}", filePath, e);
        }
        return stopwords;
    }

    private static Set<String> loadDictionary(String filePath) {
        Set<String> dictionary = new HashSet<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                dictionary.add(line.trim());
            }
        } catch (IOException e) {
            LOGGER.error("Error loading dictionary from file: {}", filePath, e);
        }
        return dictionary;
    }

    private static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        try {
            TokenStream tokenStream = new StandardAnalyzer().tokenStream(null, text);
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
                tokens.add(tokenStream.getAttribute(CharTermAttribute.class).toString());
            }
            tokenStream.close();
        } catch (IOException e) {
            LOGGER.error("Error tokenizing text: {}", text, e);
        }
        return tokens;
    }

    private static List<String> maxMatchSegment(String text) {
        List<String> segList = new ArrayList<>();
        int textLen = text.length();
        while (textLen > 0) {
            int maxLen = Math.min(7, textLen); // Limit max length to 7 characters
            boolean found = false;
            for (int i = maxLen; i > 0; i--) {
                String word = text.substring(0, i);
                if (dictionary.contains(word)) {
                    segList.add(word);
                    text = text.substring(i);
                    textLen = text.length();
                    found = true;
                    break;
                }
            }
            if (!found) {
                segList.add(text.substring(0, 1));
                text = text.substring(1);
                textLen = text.length();
            }
        }
        return segList;
    }

    private static INDArray encodeWord(String word, TokenizerFactory tokenizerFactory) {
        List<String> tokens = tokenizerFactory.create(word).getTokens();
        INDArray input = Nd4j.zeros(1, tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            input.putScalar(new int[] { 0, i }, dictionary.contains(tokens.get(i)) ? 1.0 : 0.0);
        }
        return input;
    }

    private static Map<String, INDArray> buildInvertedIndexAndWordVectors(List<String> documents,
                                                                          List<String> documentNames) {
        Map<String, SkipList> invertedIndex = new HashMap<>();
        Map<String, INDArray> wordVectors = new HashMap<>();
        for (int docId = 0; docId < documents.size(); docId++) {
            String document = documents.get(docId);
            List<String> tokens = maxMatchSegment(document);
            for (String token : tokens) {
                if (!invertedIndex.containsKey(token)) {
                    invertedIndex.put(token, new SkipList());
                    wordVectors.put(token, encodeWord(token, tokenizerFactory));
                }
                invertedIndex.get(token).insert(documentNames.get(docId), docId);
            }
        }
        return invertedIndex;
    }

    private static List<String> calculateSimilarity(String word, Map<String, INDArray> wordVectors) {
        INDArray vec1 = wordVectors.get(word);
        if (vec1 == null) {
            LOGGER.error("Word '{}' not found in vocabulary.", word);
            return null;
        }
        List<Map.Entry<String, Double>> similarities = new ArrayList<>();
        for (Map.Entry<String, INDArray> entry : wordVectors.entrySet()) {
            double similarity = cosineSimilarity.cosineSimilarity(vec1, entry.getValue());
            similarities.add(Map.entry(entry.getKey(), similarity));
        }
        similarities.sort((e1, e2) -> Double.compare(e2.getValue(), e1.getValue())); // Sort by similarity (descending)
        List<String> similarWords = new ArrayList<>();
        for (int i = 1; i < 3 && i < similarities.size(); i++) { // Retrieve top 2 similar words
            similarWords.add(similarities.get(i).getKey());
        }
        return similarWords;
    }

    private static List<String> processTextSimilarity(String documentsDirectory, String stopwordsFile) {
        List<String> documents = new ArrayList<>();
        List<String> documentNames = new ArrayList<>();
        try {
            for (String filename : new java.io.File(documentsDirectory).list()) {
                StringBuilder content = new StringBuilder();
                try (BufferedReader reader = new BufferedReader(new FileReader(documentsDirectory + "/" + filename))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        content.append(line);
                    }
                }
                documents.add(content.toString());
                documentNames.add(filename);
            }
        } catch (IOException e) {
            LOGGER.error("Error loading documents from directory: {}", documentsDirectory, e);
            return null;
        }
        Map<String, INDArray> wordVectors = buildInvertedIndexAndWordVectors(documents, documentNames);
        while (true) {
            // Input target word
            System.out.print("请输入要查询的词语（输入exit退出）：");
            String targetWord = System.console().readLine();
            if ("exit".equalsIgnoreCase(targetWord)) {
                break;
            }
            long startTime = System.currentTimeMillis();
            // Calculate similar words
            List<String> similarWords = calculateSimilarity(targetWord, wordVectors);
            if (similarWords == null) {
                continue;
            }
            // Find document IDs containing both top 2 similar words
            List<String> relevantDocIds = new ArrayList<>(wordVectors.keySet());
            for (String word : similarWords) {
                SkipList relevantDocIds1 = invertedIndex.get(word); // A SkipList
                for (String name : wordVectors.keySet()) {
                    if (!relevantDocIds1.search(name)) {
                        relevantDocIds.remove(name);
                    }
                }
            }
            long endTime = System.currentTimeMillis();
            // Output results
            System.out.println("指定词: " + targetWord);
            System.out.println("与指定词语义相似度最高的Top2个词: " + similarWords);
            System.out.println("最终输出的文档ID: " + relevantDocIds);
            System.out.println("检索时间: " + (endTime - startTime) / 1000.0 + "秒");
        }
        return null;
    }
}
