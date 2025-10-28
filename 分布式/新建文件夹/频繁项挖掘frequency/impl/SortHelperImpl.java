package DSPPCode.mapreduce.frequent_item_analysis.impl;

import DSPPCode.mapreduce.frequent_item_analysis.question.SortHelper;
import java.util.Collections;
import java.util.List;

public class SortHelperImpl extends SortHelper {
  /**
   * TODO 请完成该抽象方法
   * -
   * 输入：
   * input：待排序序列
   * 输出：
   * 按照字典序升序排好的序列，例如输入为(A,C,B)输出为(A,B,C)
   */
  @Override
  public List<String> sortSeq(List<String> input) {
    Collections.sort(input);
    return input;
  }
}
