import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
public class UDPSearcher {
    public static void main(String[] args) throws IOException {
// 1. 定义要发送的数据
//        String sendData = "⽤户名admin; 密码123";
        String sendData= MessageUtil.buildWithPort(30000);
        byte[] sendBytes = sendData.getBytes(StandardCharsets.UTF_8);
// 2. 创建发送者端的DatagramSocket对象
        DatagramSocket datagramSocket = new DatagramSocket(9092);
        datagramSocket.setBroadcast(true);
// 3. 创建数据报，包含要发送的数据
        DatagramPacket sendPacket = new DatagramPacket(sendBytes, 0, sendBytes.length,
                InetAddress.getByName("255.255.255.255"), 9091);
// 4. 向接受者端发送数据报
        datagramSocket.send(sendPacket);
        System.out.println("数据发送完毕...");
// Task 1 TODO: 准备接收Provider的回送消息; 查看接受信息并打印
        byte[] buf = new byte[1024];
        DatagramPacket receivePacket = new DatagramPacket(buf, 0, buf.length);
        System.out.println("阻塞等待接受者的回复...");
        datagramSocket.receive(receivePacket);
        int len = receivePacket.getLength();
        String data = new String(receivePacket.getData(),0, len);
        String tag=MessageUtil.parseTag(data);
        System.out.println("我是发送者，接受者回复Tag:"+ tag);
// 5. 关闭datagramSocket
        datagramSocket.close();
    }
}
class MessageUtil {
    private static final String TAG_HEADER = "special tag:";
    private static final String PORT_HEADER = "special port:";
    public static String buildWithPort(int port) {
        return PORT_HEADER + port;
    }
    public static int parsePort(String data) {
        if (data.startsWith(PORT_HEADER)) {
            return Integer.parseInt(data.substring(PORT_HEADER.length()));
        }
        return -1;
    }
    public static String buildWithTag(String tag) {
        return TAG_HEADER + tag;
    }
    public static String parseTag(String data) {
        if (data.startsWith(TAG_HEADER)) {
            return data.substring(TAG_HEADER.length());
        }
        return null;
    }
}