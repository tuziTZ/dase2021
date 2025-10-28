import java.io.*;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
public class UDPFileReceiver {
    public static void main(String[] args) throws IOException {
        File file = new File("checksum_recv.txt"); //要接收的⽂件存放路径
        FileOutputStream output = new FileOutputStream(file);
        byte[] data=new byte[1024];
        DatagramSocket ds=new DatagramSocket(9091);
        DatagramPacket dp=new DatagramPacket(data, data.length);
// TODO 实现不断接收数据报并将其通过⽂件输出流写⼊⽂件, 以数据报⻓度为零作为终⽌条件
        while(true){
            ds.receive(dp);
            int len;
            if((len=dp.getLength())==0){
                break;
            }
            System.out.println("发送数据报长度为："+len);
            String d=new String(dp.getData(),0,len);
            output.write(d.getBytes());
//            byte[] d;
//            d=dp.getData();
//            for (int i=0;i<len;i++){
//                output.write(d[i]);
//            }
        }
        output.close();
        ds.close();
        System.out.println("接收来⾃"+dp.getAddress().toString()+"的⽂件已完成！对⽅所使⽤的端⼝号为："+dp.getPort());
                file = new File("checksum_recv.txt");
        System.out.println("接收⽂件的md5为: " + MD5Util.getMD5(file));
    }
}