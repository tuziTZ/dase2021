package DSPPCode.mapreduce.common_pagerank.impl;

import DSPPCode.mapreduce.common_pagerank.question.PageRankJoinMapper;
import DSPPCode.mapreduce.common_pagerank.question.PageRankRunner;
import DSPPCode.mapreduce.common_pagerank.question.utils.Rank;
import DSPPCode.mapreduce.common_pagerank.question.utils.RanksOperation;
import DSPPCode.mapreduce.common_pagerank.question.utils.ReduceJoinWritable;
import DSPPCode.mapreduce.common_pagerank.question.utils.ReducePageRankWritable;
import com.google.common.io.Files;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class PageRankJoinMapperImpl extends PageRankJoinMapper {
  /**
   * TODO 请完成该抽象方法
   * -
   * 输入：
   * 输入数据由两个文件构成，分别是网页链接关系和网页排名，文本中的第一列都为网页名，列与列之间用空格分隔
   * <p>
   * 网页链接关系文本中的其他列为出站链接
   * 如A B D 表示网页A链向网页B和D
   * <p>
   * 网页排名文本第二列为该网页的排名值
   * 如 A 1 表示网页A的排名为1
   * <p>
   * 可借助ReduceJoinWritable类来实现
   */
  @Override
  public void map(LongWritable key, Text value,
      Mapper<LongWritable, Text, Text, ReduceJoinWritable>.Context context)
      throws IOException, InterruptedException {

    String []pageInfo=value.toString().split(" ");
    ReduceJoinWritable writableInfo;
    writableInfo=new ReduceJoinWritable();
    if(Objects.equals(pageInfo[1], "1")){

      writableInfo.setTag(ReduceJoinWritable.PAGERNAK);
      writableInfo.setData("1");
    }
    else{

      writableInfo.setTag(ReduceJoinWritable.PAGEINFO);
      StringBuilder result=new StringBuilder();

      for(int i=1;i<pageInfo.length;i++){
        result.append(pageInfo[i]).append(" ");
      }

      writableInfo.setData(result.toString());
    }
    context.write(new Text(pageInfo[0]), writableInfo);

  }
}
