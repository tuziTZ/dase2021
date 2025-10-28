package org.example;

import java.io.*;
import java.util.*;

import org.apache.commons.text.similarity.CosineSimilarity;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TextSimilarity {
    private static final Logger LOGGER = LoggerFactory.getLogger(TextSimilarity.class);

    public static void main(String[] args) {
        String documentsDirectory = "D:\\Javaprojects\\nlp41\\src\\main\\article";
        String stopwordsFile = "D:\\Javaprojects\\nlp41\\src\\main\\cn_stopwords.txt";

    }

}
