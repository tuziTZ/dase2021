package DSPPCode.mapreduce.frequent_item_analysis.impl;

import DSPPCode.mapreduce.frequent_item_analysis.question.FrequentItemAnalysisReducer;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import java.io.IOException;

public class FrequentItemAnalysisReducerImpl extends FrequentItemAnalysisReducer {
  /**
   * TODO 请完成该抽象方法
   * -
   * 输出：
   * 满足支持度的n阶频繁项，其中每一个频繁项内部的物品需要按字典序升序排列
   */
  private int supportCount;

  @Override
  protected void setup(Reducer<Text, IntWritable, Text, NullWritable>.Context context)
      throws IOException, InterruptedException {
    super.setup(context);
    int totalTransactions = context.getConfiguration().getInt("count.of.transactions", 1);
    double support = context.getConfiguration().getDouble("support", 1.0);

    supportCount = (int) Math.ceil(totalTransactions * support);
  }
  @Override
  public void reduce(Text key, Iterable<IntWritable> values,
      Reducer<Text, IntWritable, Text, NullWritable>.Context context)
      throws IOException, InterruptedException {
    int sum = 0;
    for (IntWritable value : values) {
      sum += value.get();
    }

    if (sum >= supportCount) {
      context.write(key, NullWritable.get());
    }
  }
}
