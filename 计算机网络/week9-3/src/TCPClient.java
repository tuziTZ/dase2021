import java.io.*;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

public class TCPClient {
    public static void main(String[] args) throws IOException {
        int port = 9091;
        // ⽣成并写⼊发送⽂件
        try (FileWriter fileWriter = new FileWriter("checksum.txt")) {
            Random r = new Random(2023);
// 尝试 1e3 and 1e8
            for (int i = 0; i < 1e8; i++) {
                fileWriter.write(r.nextInt());
            }
        }
        File file = new File("checksum.txt");
        System.out.println("发送⽂件⽣成完毕");
        System.out.println("发送⽂件的md5为: " + MD5Util.getMD5(file));
        FileInputStream fis = new FileInputStream(file);
        byte[] bytes = new byte[1024];
        int len;
        Socket socket = new Socket("127.0.0.1", port);
        OutputStream os = socket.getOutputStream();
        while(true){
            len = fis.read(bytes);
//            System.out.println(len);
            if(len==-1) break;
            for(int i=0;i<len;i++){
                os.write(bytes[i]);
            }
//            os.write(bytes);
        }
        byte[] a = new byte[0];
        os.write(a);
        fis.close();
        socket.close();

    }
}