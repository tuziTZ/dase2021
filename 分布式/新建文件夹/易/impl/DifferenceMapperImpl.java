package DSPPCode.mapreduce.difference.impl;

import DSPPCode.mapreduce.difference.question.DifferenceMapper;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import java.io.IOException;

public class DifferenceMapperImpl extends DifferenceMapper {

  @Override
  public void map(Object key, Text value, Mapper<Object, Text, Text, Text>.Context context)
      throws IOException, InterruptedException {
    String[] fields = value.toString().split("\t");

    context.write(new Text(fields[0]),new Text(fields[1]));

  }
}
