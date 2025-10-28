package DSPPCode.mapreduce.transitive_closure.impl;


import DSPPCode.mapreduce.transitive_closure.question.TransitiveClosureMapper;
import DSPPCode.mapreduce.transitive_closure.question.TransitiveClosureReducer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.StringTokenizer;
import java.util.regex.Pattern;


public class TransitiveClosureMapperImpl extends TransitiveClosureMapper{


  @Override
  public void map(Object key, Text value, Mapper<Object, Text, Text, Text>.Context context)
      throws IOException, InterruptedException {

    if (Objects.equals(value.toString(), "child\tparent")){}
    else{
      String[] tokens = value.toString().split(" ");
      if (tokens.length == 2) {
        Text child = new Text(tokens[0]);
        Text parent = new Text(tokens[1]);
        context.write(child, parent);
      }

    }


    }




  }

