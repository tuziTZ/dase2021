package DSPPCode.mapreduce.frequent_item_analysis.impl;

import DSPPCode.mapreduce.frequent_item_analysis.question.FrequentItemAnalysisMapper;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class FrequentItemAnalysisMapperImpl extends FrequentItemAnalysisMapper {

  private List<Set<String>> generateCandidateItemSets(Set<String> items, int n) {
    List<Set<String>> candidateSets = new ArrayList<>();

    if (n == 1) {
      for (String item : items) {
        Set<String> set = new HashSet<>();
        set.add(item);
        candidateSets.add(set);
      }
      return candidateSets;
    }



    List<Set<String>> prevCandidates = generateCandidateItemSets(items, n - 1);
    Map<String, Integer> itemSupportCount = getItemSupportCount(prevCandidates);

    for (int i = 0; i < prevCandidates.size(); i++) {
      for (int j = i + 1; j < prevCandidates.size(); j++) {
        Set<String> set1 = prevCandidates.get(i);
        Set<String> set2 = prevCandidates.get(j);

        Set<String> unionSet = new HashSet<>(set1);
        unionSet.addAll(set2);

        if (unionSet.size() == n && isSubsetFrequent(unionSet, prevCandidates, itemSupportCount)) {
          candidateSets.add(unionSet);
        }
      }
    }

    return candidateSets;
  }

  private Map<String, Integer> getItemSupportCount(List<Set<String>> itemSets) {
    Map<String, Integer> supportCount = new HashMap<>();
    for (Set<String> itemSet : itemSets) {
      for (String item : itemSet) {
        supportCount.put(item, supportCount.getOrDefault(item, 0) + 1);
      }
    }
    return supportCount;
  }

  private boolean isSubsetFrequent(Set<String> itemSet, List<Set<String>> prevCandidates, Map<String, Integer> itemSupportCount) {
    for (String item : itemSet) {
      Set<String> subset = new HashSet<>(itemSet);
      subset.remove(item);
      if (!prevCandidates.contains(subset) || itemSupportCount.get(item) < itemSupportCount.get(subset.iterator().next())) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void map(LongWritable key, Text value,
      Mapper<LongWritable, Text, Text, IntWritable>.Context context)
      throws IOException, InterruptedException {
    String[] items = value.toString().split(",");
    int n = context.getConfiguration().getInt("number.of.pairs", 1);

    Set<String> itemSet = new HashSet<>(Arrays.asList(items));
    List<Set<String>> candidateSets = generateCandidateItemSets(itemSet, n);

    for (Set<String> candidateSet : candidateSets) {
      String[] sortedItems = candidateSet.toArray(new String[0]);
      Arrays.sort(sortedItems);
      String itemString = String.join(",", sortedItems);
      context.write(new Text(itemString), new IntWritable(1));
    }
  }
}
