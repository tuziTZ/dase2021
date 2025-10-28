package DSPPCode.mapreduce.transitive_closure.impl;


import DSPPCode.mapreduce.transitive_closure.question.TransitiveClosureMapper;
import DSPPCode.mapreduce.transitive_closure.question.TransitiveClosureReducer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.StringTokenizer;
import java.util.regex.Pattern;


public class TransitiveClosureMapperImpl extends TransitiveClosureMapper {


  @Override
  public void map(Object key, Text value, Mapper<Object, Text, Text, Text>.Context context)
      throws IOException, InterruptedException {

    if (Objects.equals(value.toString(), "child\tparent")) {
    } else {
      String line = value.toString();
      String[] childAndParent = line.split(" ");
      List<String> list = new ArrayList<>(2);
      for (String childOrParent : childAndParent) {
        if (!"".equals(childOrParent)) {
          list.add(childOrParent);
        }
      }
      if (!"child".equals(list.get(0))) {
        String childName = list.get(0);
        String parentName = list.get(1);
        String relationType = "1";
        context.write(new Text(parentName), new Text(relationType + "+"
            + childName + "+" + parentName));
        relationType = "2";
        context.write(new Text(childName), new Text(relationType + "+"
            + childName + "+" + parentName));


      }


    }


  }
}

