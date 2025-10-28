import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

public class UDPProvider {
    public static void main(String[] args) throws IOException {
// 1. 创建接受者端的DatagramSocket，并指定端⼝
        DatagramSocket datagramSocket = new DatagramSocket(9091);
// 2. 创建数据报，⽤于接受客户端发来的数据
        byte[] buf = new byte[1024];
        DatagramPacket receivePacket = new DatagramPacket(buf, 0, buf.length);
// 3. 接受客户端发送的数据，此⽅法在收到数据报之前会⼀直阻塞
        System.out.println("阻塞等待发送者的消息...");
        datagramSocket.receive(receivePacket);
// 4. 解析数据
        String ip = receivePacket.getAddress().getHostAddress();
        int port = receivePacket.getPort();
        int len = receivePacket.getLength();
        String data = new String(receivePacket.getData(),0, len);
        int pt=MessageUtil.parsePort(data);
        System.out.println("我是接受者，" + ip + ":" + port + " 的发送者输出的端口号是: "+ pt);
// Task 1 TODO: 准备回送数据; 创建数据报，⽤于发回给发送端; 发送数据报
        String tag = UUID.randomUUID().toString();
        String newData=MessageUtil.buildWithTag(tag);
        byte[] sendBytes = newData.getBytes(StandardCharsets.UTF_8);
        DatagramPacket returnPacket = new DatagramPacket(sendBytes,0,sendBytes.length,
                InetAddress.getLocalHost(), 9092);
        datagramSocket.send(returnPacket);
        System.out.println("数据回复完毕...");

// 5. 关闭datagramSocket
        datagramSocket.close();
    }
}
