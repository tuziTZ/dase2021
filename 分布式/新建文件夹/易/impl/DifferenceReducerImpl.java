package DSPPCode.mapreduce.difference.impl;

import DSPPCode.mapreduce.difference.question.DifferenceReducer;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import scala.xml.Null;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class DifferenceReducerImpl extends DifferenceReducer {

  @Override
  public void reduce(Text key, Iterable<Text> values,
      Reducer<Text, Text, Text, NullWritable>.Context context)
      throws IOException, InterruptedException {
    Map<String, Integer> counterMap = new HashMap<>();
    for(Text value:values){
      counterMap.put(value.toString(), counterMap.getOrDefault(value.toString(), 0) + 1);
    }

    for (String k : counterMap.keySet()) {
      Integer value = counterMap.get(k);
      if(value!=2){
        String result=key.toString()+'\t'+k;
        context.write(new Text(result),null);
      }

    }



    // Set<String> set = new TreeSet<String>();
    // for(Text tex : values){
    //   if (set.contains(tex.toString())) {
    //     set.remove(tex.toString());
    //     String result=key.toString()+'\t'+ tex;
    //     context.write(new Text(result),null);
    //   }
    //   else{
    //     set.add(tex.toString());
    //   }
    //
    // }
    // for(String tex : set){
    //   String result=key.toString()+'\t'+ tex;
    //   context.write(new Text(result),null);
    // }





  }
}
