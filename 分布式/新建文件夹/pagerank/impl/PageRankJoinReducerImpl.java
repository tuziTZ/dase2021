package DSPPCode.mapreduce.common_pagerank.impl;

import DSPPCode.mapreduce.common_pagerank.question.PageRankJoinReducer;
import DSPPCode.mapreduce.common_pagerank.question.PageRankRunner;
import DSPPCode.mapreduce.common_pagerank.question.utils.ReduceJoinWritable;
import DSPPCode.mapreduce.common_pagerank.question.utils.ReducePageRankWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import java.io.IOException;
import java.util.Arrays;

public class PageRankJoinReducerImpl extends PageRankJoinReducer {
  private static final double D=0.85;
  /**
   * TODO 请完成该抽象方法
   *
   * <p>输出： 输出文本为网页链接关系和网页排名连接后的结果
   *
   * <p>如 A 1.0 B D 表示网页A的排名为1，并且链向网页B和D (题目中网页权重均按1.0计算)
   *
   * <p>可借助ReduceJoinWritable类来实现
   */
  @Override
  public void reduce(Text key, Iterable<ReduceJoinWritable> values,
      Reducer<Text, ReduceJoinWritable, Text, NullWritable>.Context context)
      throws IOException, InterruptedException {

    String[] pageInfo = new String[0];
    String pageRank = null;
    StringBuilder result=new StringBuilder();
    for(ReduceJoinWritable value:values){
      String tag=value.getTag();
//            如果是贡献值则求和，否则以空格为分隔符切分后保存到pageInfo
      if(tag.equals(ReduceJoinWritable.PAGEINFO)){
        pageInfo=value.getData().split(" ");

      }else if(tag.equals(ReduceJoinWritable.PAGERNAK)){
        pageRank=value.getData();
      }
    }
    result.append(key).append(" ").append(pageRank).append(" ");

    for(String info : pageInfo){
      result.append(info).append(" ");
    }

    context.write(new Text(result.toString()), NullWritable.get());

  }
}
