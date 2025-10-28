import java.io.*;
import java.util.*;

public class DictionaryCompressor {

    // 定义块大小，可以根据实际情况调整
    private static final int BLOCK_SIZE = 4;
    private int blockLength=0;
//    private int offsetLength=0;
    //存储前缀后缀长度所需要的字符数
    private static int suffixLength=0;
    private static int prefixLength=0;

    public static void main(String[] args) throws IOException {
        Map<String, String> dictionary = readDictionary("dict.txt");
        long srcLength=calculateDictionarySize("dict.txt");
        System.out.println("dict file length:"+srcLength);
//        System.out.println(terms.get(0));



        List<String> compressedDictionary = compressDictionary(new ArrayList<>(dictionary.keySet()), dictionary);
        writeCompressedDictionary(compressedDictionary, "compressed_dict.txt");
        long dictLength=calculateDictionarySize("compressed_dict.txt");
        System.out.println("compressed_dict file length:"+dictLength);
        List<String> decompressedDictionary = decompressDictionary("compressed_dict.txt");
        writeCompressedDictionary(decompressedDictionary, "decompressed_dict.txt");
        System.out.println("decompressed_dict file length:"+calculateDictionarySize("decompressed_dict.txt"));
        System.out.println("compression ratio:"+(float)dictLength/(float)srcLength);

    }
    private static long calculateDictionarySize(String fileName) throws IOException {
        File file = new File(fileName);
        if (!file.exists() || !file.isFile()) {
            throw new FileNotFoundException("File not found or is not a valid file: " + fileName);
        }
        return file.length();
    }

    // 读取词典文件
    private static Map<String, String> readDictionary(String fileName) throws IOException {
        Map<String, String> dictionary = new TreeMap<>(); // 使用 TreeMap 自动按字典序排列
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line;
        while ((line = reader.readLine()) != null) {
            // 解析每行中的所有词项数据
            int length = line.length();
            for (int i = 0; i < length; i += 115) {
                if (i + 115 > length) break; // 防止越界
                String term = line.substring(i, i + 107).trim();
                String pointer = line.substring(i + 107, i + 115).trim();
                dictionary.put(term, pointer);
            }
        }
        reader.close();

        return dictionary;
    }
    //块：块内所有字符总长度+每个词项的块内位移+词项的公共前缀+各个词项的数据
    //每个词项的块内位移

    //词项：词项的长度（仅后缀部分）+词项后缀+倒排索引表指针
    //块长度blockLength 偏移1 偏移2 偏移3 偏移4 前缀长度 后缀长度1234
    //45 0 11 23 33 4 norc 2 ia 23000817 3 ini 23393816 1 o 23393823 3 ott 21941434

    private static List<String> compressDictionary(List<String> terms, Map<String, String> dictionary) {
        List<List<String>> compressedDictionary = new ArrayList<>();
        for (int i = 0; i < terms.size(); i += BLOCK_SIZE) {
            int end = Math.min(i + BLOCK_SIZE, terms.size());
            List<String> block = terms.subList(i, end);
            compressedDictionary.add(compressBlock(block, dictionary));
        }
        List<String> blockList=new ArrayList<>();
        //根据已知的前缀后缀长度，计算记录块长度所需要的字符数，并建立，其中第一个数字是前缀，其它字符串前的数字表示后缀
        for(int i=0;i<compressedDictionary.size();i++){
            StringBuilder blockData = new StringBuilder();
            List<String> tempList=compressedDictionary.get(i);
//            prefixLength\suffixLength
            //eg:[1, é, 5, toile23501351, 4, tude23096305, 5, tudes22198822, 5, ulard23501358]
            String prefix=tempList.get(0);
            String paddedPrefix = String.format("%0" + prefixLength + "d", Integer.parseInt(prefix));
            blockData.append(paddedPrefix).append(tempList.get(1));
            for(int j=1;j<BLOCK_SIZE+1;j++){
                String suffix=tempList.get(j*2);
                String paddedSuffix = String.format("%0" + suffixLength + "d", Integer.parseInt(suffix));
                blockData.append(paddedSuffix).append(tempList.get(j*2+1));
            }
            blockList.add(blockData.toString());
//            System.out.println(blockData);
            //01 é 005 toile 23501351 004 tude 23096305 005 tudes 22198822
        }
        //构建块，最终返回多个块组成的字符串
        return blockList;
    }

    private static List<String> compressBlock(List<String> block, Map<String, String> dictionary) {

        List<String> compressedBlock = new ArrayList<>();
        List<Integer> offsets = new ArrayList<>();
//        StringBuilder termsData = new StringBuilder();
//        int currentOffset = 0;

        // 块内共同前缀长度
        int commonPrefixLength = commonPrefixLength(block);
        // 块内共同前缀
        String commonPrefix = block.get(0).substring(0, commonPrefixLength);
        //写入前缀信息
        compressedBlock.add(String.valueOf(commonPrefix.length()));
        if(prefixLength<String.valueOf(commonPrefix.length()).length()){
            prefixLength=String.valueOf(commonPrefix.length()).length();
        }
        compressedBlock.add(commonPrefix);

        for (String term : block) {
            StringBuilder termsData = new StringBuilder();
            //词后缀
            String suffix = term.substring(commonPrefixLength);
            //倒排索引文件指针
            String pointer = dictionary.get(term);
            //offsets表示每个词后缀开始的位置
//            offsets.add(currentOffset);
            if(suffixLength<String.valueOf(suffix.length()).length()){
                suffixLength=String.valueOf(suffix.length()).length();
            }
            termsData.append(suffix).append(pointer);
            compressedBlock.add(String.valueOf(suffix.length()));
            compressedBlock.add(termsData.toString());
//            currentOffset += Integer.toString(suffix.length()).length() + suffix.length() + pointer.length();
        }
        //用特定几位存储块总长度
        //确定位数之后，每个offset的位数也确定了

//        compressedBlock.add(String.valueOf(currentOffset));
//        for (int offset : offsets) {
//            compressedBlock.add(String.valueOf(offset));
//
//        }

//        System.out.println(compressedBlock);

        return compressedBlock;
    }

    private static int commonPrefixLength(List<String> strings) {
        if (strings == null || strings.isEmpty()) {
            return 0;
        }
        String first = strings.get(0);
        int prefixLength = first.length();

        for (int i = 1; i < strings.size(); i++) {
            prefixLength = commonPrefixLength(first, strings.get(i), prefixLength);
            if (prefixLength == 0) {
                break;
            }
        }

        return prefixLength;
    }

    private static int commonPrefixLength(String s1, String s2, int maxLength) {
        int minLength = Math.min(Math.min(s1.length(), s2.length()), maxLength);
        for (int i = 0; i < minLength; i++) {
            if (s1.charAt(i) != s2.charAt(i)) {
                return i;
            }
        }
        return minLength;
    }

    private static void writeCompressedDictionary(List<String> compressedDictionary, String fileName) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        for (String line : compressedDictionary) {
            writer.write(line);
//            writer.newLine();
        }
        writer.close();
    }

    private static List<String> decompressDictionary(String fileName) throws IOException {
        List<String> decompressedDictionary = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        StringBuilder allData = new StringBuilder();
        String line;

        // Read all lines and concatenate them into a single string
        while ((line = reader.readLine()) != null) {
            allData.append(line);
        }
        reader.close();

        String data = allData.toString();
        int index = 0;


        while (index < data.length()) {

            int commonPrefixLength = Integer.parseInt(data.substring(index, index + prefixLength));
            index += prefixLength;
            String prefix=data.substring(index, index + commonPrefixLength);
            index += commonPrefixLength;

            for (int i=0;i<BLOCK_SIZE;i++){
                int suffixLength0 = Integer.parseInt(data.substring(index, index + suffixLength));
                index += suffixLength;
                String suffix = data.substring(index, index + suffixLength0);
                index += suffixLength0;

                // 指针长度为8
                String pointer = data.substring(index, index + 8);
                index += 8;

                String word = prefix + suffix;

                word = String.format("%107s", word);

//                pointer = String.format("%08d", Integer.parseInt(pointer));

                String decompressedEntry = word + pointer;
                decompressedDictionary.add(decompressedEntry);
            }


        }

        return decompressedDictionary;
    }



}


